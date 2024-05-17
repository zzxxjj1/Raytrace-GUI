import numpy as cp
# import cupy as cp #tried to use cupy for accalerate the rendering speed but the newest version of cupy have a directory related bug on Windows
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk  # 这里添加 ttk 的导入
from tkinter import simpledialog
from multiprocessing import Process, Array, Value, Lock
import time

camera = cp.array([0, 0, 1])
        
def render_part(arr, progress_counter, lock, total_rows, y_start, y_end, width, height, camera, light_position, spheres, ambient, kd, ks):
    image = cp.frombuffer(arr.get_obj()).reshape((height, width, 3))
    for y in range(y_start, y_end):
        for x in range(width):
            px = (x - width / 2) / (width / 2)
            py = -(y - height / 2) / (height / 2)
            ray_direction = normalize(cp.array([px, py, -1]))
            color = trace_ray(camera, ray_direction, spheres=spheres, light_position=light_position, ambient=ambient, kd=kd, ks=ks)
            image[y, x] = cp.clip(color * 255, 0, 255)
        with lock:
            progress_counter.value += 1  # 更新进度

def render_image_multiprocessed():
    width, height = 512, 512
    camera = cp.array([0, 0, 1])
    shared_array = Array('d', height * width * 3)  # 创建共享内存数组
    image = cp.frombuffer(shared_array.get_obj()).reshape((height, width, 3))

    # Serialize spheres data for multiprocessing
    spheres_data = [(sphere['center'].tolist(), sphere['radius'], sphere['color'].tolist(), sphere['reflectivity']) for sphere in spheres]

    processes = []
    num_processes = 4  # 根据CPU核心数调整
    rows_per_process = height // num_processes

    # Include ambient, kd, and ks in the args for each process
    for i in range(num_processes):
        y_start = i * rows_per_process
        y_end = height if i == num_processes - 1 else (i + 1) * rows_per_process
        proc = Process(target=render_part, args=(shared_array, y_start, y_end, width, height, camera, light_position, spheres_data, ambient, kd, ks))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    return Image.fromarray(image.astype('uint8'), 'RGB')

def normalize(v):
    return v / cp.linalg.norm(v)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * cp.dot(ray_direction, ray_origin - center)
    c = cp.dot(ray_origin - center, ray_origin - center) - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b - cp.sqrt(delta)) / 2
        t2 = (-b + cp.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def trace_ray(ray_origin, ray_direction, depth=0, spheres=None, light_position=None, ambient=None, kd=None, ks=None):
    if depth > 3:
        return cp.array([0.9, 0.9, 0.9])

    min_t = cp.inf
    nearest_sphere = None
    for center, radius, color, reflectivity in spheres:
        center = cp.array(center)
        color = cp.array(color)
        t = sphere_intersect(center, radius, ray_origin, ray_direction)
        if t is not None and t < min_t:
            min_t = t
            nearest_sphere = {'center': center, 'radius': radius, 'color': color, 'reflectivity': reflectivity}
    if nearest_sphere is None:
        return cp.array([0.9, 0.9, 0.9])

    hit_point = ray_origin + min_t * ray_direction
    normal = normalize(hit_point - nearest_sphere['center'])
    light_direction = normalize(light_position - hit_point)
    light_intensity = max(0, cp.dot(light_direction, normal))
    view_direction = normalize(ray_origin - hit_point)

    if is_in_shadow(hit_point, light_position, spheres):
        light_intensity = 0
    else:
        light_intensity = max(0, cp.dot(light_direction, normal))

    diffuse = light_intensity * kd
    reflection = max(0, cp.dot(light_direction, 2 * normal * cp.dot(normal, view_direction) - view_direction))
    specular = ks * reflection ** 2

    color = nearest_sphere['color'] * (ambient + diffuse) + cp.array([0.9, 0.9, 0.9]) * specular
    
    if 'reflectivity' in nearest_sphere:
        reflect_direction = ray_direction - 2 * cp.dot(ray_direction, normal) * normal
        reflect_color = trace_ray(hit_point, reflect_direction, depth + 1, spheres=spheres, light_position=light_position, ambient=ambient, kd=kd, ks=ks)
        color = color * (1 - nearest_sphere['reflectivity']) + reflect_color * nearest_sphere['reflectivity']

    return color

def update_sphere(i, new_center=None, new_radius=None, new_color=None, new_reflectivity=None):
    if new_center is not None:
        spheres[i]['center'] = cp.array(new_center)
    if new_radius is not None:
        spheres[i]['radius'] = float(new_radius)
    if new_color is not None:
        spheres[i]['color'] = cp.array(new_color)
    if new_reflectivity is not None:
        spheres[i]['reflectivity'] = float(new_reflectivity)
    update_canvas()



def is_in_shadow(test_point, light_position, spheres):
    direction = normalize(light_position - test_point)
    for sphere in spheres:
        center = cp.array(sphere[0])
        radius = sphere[1]
        if sphere_intersect(center, radius, test_point, direction) is not None:
            return True
    return False

def update_canvas():
    global camera  
    width, height = 512, 512
    progress_counter = Value('i', 0)
    lock = Lock()
    progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate', maximum=height)
    progress.pack()
    root.update()

    shared_array = Array('d', height * width * 3)
    spheres_data = [(sphere['center'].tolist(), sphere['radius'], sphere['color'].tolist(), sphere['reflectivity']) for sphere in spheres]

    processes = []
    rows_per_process = height // 4
    start_time = time.time()  # Start timing

    for i in range(4):
        y_start = i * rows_per_process
        y_end = height if i == 3 else (i + 1) * rows_per_process
        proc = Process(target=render_part, args=(shared_array, progress_counter, lock, height, y_start, y_end, width, height, camera, light_position, spheres_data, ambient, kd, ks))
        processes.append(proc)
        proc.start()

    while any(p.is_alive() for p in processes):
        with lock:
            progress['value'] = progress_counter.value
            root.update()
        time.sleep(0.1)

    for proc in processes:
        proc.join()

    end_time = time.time()  # End timing
    rendering_time = end_time - start_time  # Calculate total time taken
    print('rendering_time: ', rendering_time)

    img = Image.fromarray(cp.frombuffer(shared_array.get_obj()).reshape((height, width, 3)).astype('uint8'), 'RGB')
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.image = photo
    #status_label.config(text="Render complete")
    progress.pack_forget()


def edit_sphere_properties(i):
    def submit():
        new_center = [float(e_center_x.get()), float(e_center_y.get()), float(e_center_z.get())]
        new_color = [float(e_color_r.get()), float(e_color_g.get()), float(e_color_b.get())]
        update_sphere(i, new_center=new_center, new_radius=e_radius.get(), new_color=new_color, new_reflectivity=e_reflectivity.get())
        edit_window.destroy()

    edit_window = tk.Toplevel(root)
    edit_window.title(f"Edit Sphere {i+1}")

    tk.Label(edit_window, text="Center X:").grid(row=0, column=0)
    e_center_x = tk.Entry(edit_window)
    e_center_x.insert(0, spheres[i]['center'][0])
    e_center_x.grid(row=0, column=1)

    tk.Label(edit_window, text="Center Y:").grid(row=1, column=0)
    e_center_y = tk.Entry(edit_window)
    e_center_y.insert(0, spheres[i]['center'][1])
    e_center_y.grid(row=1, column=1)

    tk.Label(edit_window, text="Center Z:").grid(row=2, column=0)
    e_center_z = tk.Entry(edit_window)
    e_center_z.insert(0, spheres[i]['center'][2])
    e_center_z.grid(row=2, column=1)

    tk.Label(edit_window, text="Radius:").grid(row=3, column=0)
    e_radius = tk.Scale(edit_window, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL)
    e_radius.set(spheres[i]['radius'])
    e_radius.grid(row=3, column=1)

    tk.Label(edit_window, text="Color R:").grid(row=4, column=0)
    e_color_r = tk.Entry(edit_window)
    e_color_r.insert(0, spheres[i]['color'][0])
    e_color_r.grid(row=4, column=1)

    tk.Label(edit_window, text="Color G:").grid(row=5, column=0)
    e_color_g = tk.Entry(edit_window)
    e_color_g.insert(0, spheres[i]['color'][1])
    e_color_g.grid(row=5, column=1)

    tk.Label(edit_window, text="Color B:").grid(row=6, column=0)
    e_color_b = tk.Entry(edit_window)
    e_color_b.insert(0, spheres[i]['color'][2])
    e_color_b.grid(row=6, column=1)

    tk.Label(edit_window, text="Reflectivity:").grid(row=7, column=0)
    e_reflectivity = tk.Scale(edit_window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
    e_reflectivity.set(spheres[i]['reflectivity'])
    e_reflectivity.grid(row=7, column=1)

    btn_submit = tk.Button(edit_window, text="Submit", command=submit)
    btn_submit.grid(row=8, columnspan=2)


def edit_camera_properties():
    def submit():
        global camera
        camera = cp.array([float(e_camera_x.get()), float(e_camera_y.get()), float(e_camera_z.get())])
        edit_window.destroy()
        update_canvas()

    edit_window = tk.Toplevel(root)
    edit_window.title("Edit Camera Properties")

    tk.Label(edit_window, text="Camera Position X:").grid(row=0, column=0)
    e_camera_x = tk.Entry(edit_window)
    e_camera_x.insert(0, camera[0])
    e_camera_x.grid(row=0, column=1)

    tk.Label(edit_window, text="Camera Position Y:").grid(row=1, column=0)
    e_camera_y = tk.Entry(edit_window)
    e_camera_y.insert(0, camera[1])
    e_camera_y.grid(row=1, column=1)

    tk.Label(edit_window, text="Camera Position Z:").grid(row=2, column=0)
    e_camera_z = tk.Entry(edit_window)
    e_camera_z.insert(0, camera[2])
    e_camera_z.grid(row=2, column=1)

    btn_submit = tk.Button(edit_window, text="Submit", command=submit)
    btn_submit.grid(row=3, columnspan=2)

def edit_light_and_material_properties():
    def submit():
        global light_position, ambient, ks, kd
        light_position = cp.array([float(e_light_x.get()), float(e_light_y.get()), float(e_light_z.get())])
        ambient = float(e_ka.get())
        ks = float(e_ks.get())
        kd = float(e_kd.get())
        edit_window.destroy()
        update_canvas()

    edit_window = tk.Toplevel(root)
    edit_window.title("Edit Light and Material Properties")

    tk.Label(edit_window, text="Light Position X:").grid(row=0, column=0)
    e_light_x = tk.Entry(edit_window)
    e_light_x.insert(0, light_position[0])
    e_light_x.grid(row=0, column=1)

    tk.Label(edit_window, text="Light Position Y:").grid(row=1, column=0)
    e_light_y = tk.Entry(edit_window)
    e_light_y.insert(0, light_position[1])
    e_light_y.grid(row=1, column=1)

    tk.Label(edit_window, text="Light Position Z:").grid(row=2, column=0)
    e_light_z = tk.Entry(edit_window)
    e_light_z.insert(0, light_position[2])
    e_light_z.grid(row=2, column=1)

    tk.Label(edit_window, text="Ambient Coefficient (ka):").grid(row=3, column=0)
    e_ka = tk.Scale(edit_window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
    e_ka.set(ambient)
    e_ka.grid(row=3, column=1)

    tk.Label(edit_window, text="Specular Coefficient (ks):").grid(row=4, column=0)
    e_ks = tk.Scale(edit_window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
    e_ks.set(ks)
    e_ks.grid(row=4, column=1)

    tk.Label(edit_window, text="Diffuse Coefficient (kd):").grid(row=5, column=0)
    e_kd = tk.Scale(edit_window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
    e_kd.set(kd)
    e_kd.grid(row=5, column=1)

    btn_submit = tk.Button(edit_window, text="Submit", command=submit)
    btn_submit.grid(row=6, columnspan=2)

def render_image():
    width, height = 512, 512
    camera = cp.array([0, 0, 1])
    light_position = cp.array([-5, 5, 5])
    image = cp.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            px = (x - width / 2) / (width / 2)
            py = -(y - height / 2) / (height / 2)
            ray_direction = normalize(cp.array([px, py, -1]))
            color = trace_ray(camera, ray_direction)
            image[y, x] = cp.clip(color * 255, 0, 255)
    # image = cp.asnumpy(image)
    return Image.fromarray(image.astype('uint8'), 'RGB')

if __name__ == "__main__":
    light_position = cp.array([0, 5, 0])
    ambient = 0.50 # ambient coefficient
    ks = 0.50  # Specular coefficient
    kd = 0.70  # Diffuse coefficient

    spheres = [
        {'center': cp.array([-3, -1, -4]), 'radius': 1.5, 'color': cp.array([0.5, 1, 0.5]), 'reflectivity': 0.2},
        {'center': cp.array([0, 2, -4]), 'radius': 1.5, 'color': cp.array([1, 0, 0]), 'reflectivity': 0.2},
        {'center': cp.array([3, 0, -4]), 'radius': 1.5, 'color': cp.array([1, 1, 0]), 'reflectivity': 0.2},
        {'center': cp.array([0, -14, -5]), 'radius': 12, 'color': cp.array([0.2, 0.2, 0.2]), 'reflectivity': 0.9}
    ]
    root = tk.Tk()
    root.title("Ray Tracing Editor")

    canvas = tk.Canvas(root, width=512, height=512)
    canvas.pack()

    update_button = tk.Button(root, text="Render", command=update_canvas)
    update_button.pack()

    # status_label = tk.Label(root, text="Rendering")
    # status_label.pack()
    
    for i, sphere in enumerate(spheres[:-1]):
        btn_edit = tk.Button(root, text=f"Edit Sphere {i+1}", command=lambda i=i: edit_sphere_properties(i))
        btn_edit.pack()

    edit_light_button = tk.Button(root, text="Edit Light and Material Properties", command=edit_light_and_material_properties)
    edit_light_button.pack()

    edit_camera_button = tk.Button(root, text="Edit Camera Properties", command=edit_camera_properties)
    edit_camera_button.pack()

    update_canvas()  # Initial rendering
    root.mainloop()