# Imports
#---------

import os
import time
import cv2
import tempfile
import tkFileDialog
import tempfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.animation
import matplotlib.image as mpimg
import Tkinter as tk
from Tkinter import *
from pykinect import nui
from PIL import Image, ImageTk, ImageDraw, ImageFont
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from osgeo import gdal, osr

#------------------------------------------------------------------------------------------------------------#
# Global Variables 
#------------------

kinect = None
global_depth_data = None
capture_queue = []
# pts1 = np.zeros((20, 2), dtype=np.float32)
# pts2 = np.zeros((20, 2), dtype=np.float32)
matplotlib.use('TkAgg')
generated_checkerboard_dir = os.path.abspath('generated_checkerboard_URD')
capture_output_dir = os.path.abspath('Tfou')
interpolated_depth = None
interpolated_depth_plot = None
animation_running = False
substraction_running = False
mnt = None
sub_data = None
matrix_file = None
comparison = None
comparison_matrix =None
matrix = None
dem = None
pts1 = np.array(
                [[10, 10], [10, 580], [80, 100], [150, 250], [150, 450], [220, 120], [235, 295], [270, 50], [300, 270],
                 [300, 470], [460, 10], [460, 580]], dtype=np.float32)
pts2 = np.array(
                [[68, 58], [76, 530], [126, 134], [173, 250], [181, 422], [230, 148], [243, 294], [267, 91], [293, 273],
                 [295, 436], [407, 58], [416, 524]], dtype=np.float32)



#------------------------------------------------------------------------------------------------------------#
# Fucntions 
#-----------

def is_kinect_connected():
    return kinect is not None

def connect_kinect():
    global kinect
    try:
        status_label.config(text="Connecting...", font=('Roboto',10))
        root.update_idletasks()  # Force update to show "Connecting..."
        kinect = nui.Runtime()
        kinect.skeleton_engine.enabled = True
        status_label.config(text="Kinect Connected", font=('Roboto',10))
    except Exception as e:
        print("Error:", e)
        status_label.config(text="Kinect Not Connected", font=('Roboto',10))

def get_depth_mage(frame):
    global global_depth_data, depth_plot, depth_image_label
    try:
        depth_data = np.zeros((480, 640), dtype=np.uint16)
        frame.image.copy_bits(depth_data.ctypes.data)
        depth_data = np.asarray(depth_data, dtype=float)
        depth_data = depth_data/8
        depth_data = depth_data[0:472, 8:620]
        depth_data = gaussian_filter(depth_data, sigma=7)
        m = depth_data.max()
        depth_data = m-depth_data
        a = depth_data.max()
        global_depth_data = depth_data
        depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_data.astype(np.uint8), cv2.COLORMAP_JET)

        # Convert OpenCV image to PIL Image
        depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        depth_image = Image.fromarray(depth_image)
        
        # Resize image to fit the canvas
        depth_image = depth_image.resize((550, 420), Image.ANTIALIAS)

        # Convert PIL Image to Tkinter PhotoImage
        depth_image_tk = ImageTk.PhotoImage(depth_image)

        # Update the canvas with the new image
        canvas.create_image(0, 0, anchor=tk.NW, image=depth_image_tk)
        canvas.image = depth_image_tk  # Keep a reference to avoid garbage collection

    except Exception as e:
        print("Error:", e)

def display_depth():
    if is_kinect_connected():
        try:
            kinect.depth_frame_ready += get_depth_mage
            kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)
            pass
        except Exception as e:
            print("Error:", e)
    else:
        status_label.config(text="Kinect Not Connected", font=('Roboto',10))

def project_checkerboard_images_and_capture(output_dir, capture_output_dir, kinect):
    global capture_queue

    # Ensuring the output directory exists
    if not os.path.exists(output_dir):
        print("Error: Output directory does not exist:", output_dir)
        return

    # Creating a temporary directory for PPM files
    temp_dir = tempfile.mkdtemp()

    # Getting list of checkerboard images and sorting them
    image_files = os.listdir(output_dir)
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    def project_next_image():
        global capture_queue

        if not capture_queue:
            # Cleanup the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print("Finished projecting all images.")
            return

        index = capture_queue.pop(0)
        if index >= len(image_files):
            print("Invalid index: {}, skipping.".format(index))
            project_next_image()
            return

        image_file = image_files[index]
        image_path = os.path.join(output_dir, image_file)

        # Converting image to PPM format
        ppm_image_path = convert_to_ppm(image_path, temp_dir)
        if ppm_image_path is None:
            project_next_image()
            return

        # Displaying the image using matplotlib
        fig = plt.figure(figsize=(1300 / 100, 800 / 100))
        fig.canvas.manager.window.wm_geometry("+2000+-0")
        fig.canvas.manager.window.overrideredirect(1)
        fig.canvas.manager.window.state('zoomed')
        hAx = plt.gca()
        hAx.set_position([0, 0, 1, 1])
        plt.axis('off')
        image = mpimg.imread(ppm_image_path)

        # Displaying the image
        plt.imshow(image)
        plt.axis('auto')  #delay for projection
        plt.pause(1)  

        # Capturing and storing the image
        capture_and_store_image(fig, ppm_image_path, image_file)

    def capture_and_store_image(fig, ppm_image_path, image_file):
        global capture_queue

        attempt = len(image_files) - len(capture_queue)
        img_path = capture_color_image(kinect, capture_output_dir, attempt)
        if img_path:
            captured_image = cv2.imread(img_path)

            # Rotating the image to the right twice
            rotated_image = cv2.rotate(captured_image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)

            # Mirroring the image
            mirrored_image = cv2.flip(rotated_image, 1)

            processed_img_path = os.path.join(capture_output_dir, img_path)
            cv2.imwrite(processed_img_path, mirrored_image)

            print("Processed image saved to:", processed_img_path)
        else:
            print("Failed to capture image with Kinect.")

        plt.close(fig)

        # Projecting the next image after capturing
        project_next_image()

    capture_queue = list(range(len(image_files)))
    project_next_image()

    # Waiting until capture queue is empty
    while capture_queue:
        time.sleep(1)  # Adjust the delay time if needed

    print("All images projected, captured, processed, and saved.")

def convert_to_ppm(image_path, temp_dir):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open or find the image.")
        return None

    ppm_image_path = os.path.join(temp_dir, os.path.basename(os.path.splitext(image_path)[0] + ".ppm"))
    cv2.imwrite(ppm_image_path, img)
    return ppm_image_path

def capture_color_image(kinect, capture_output_dir, attempt):
    try:
        # Ensuring the capture output directory exists
        if not os.path.exists(capture_output_dir):
            os.makedirs(capture_output_dir)

        if is_kinect_connected():
            frame = None
            max_attempts = 10
            current_attempt = 0

            # Waiting for the Kinect to stabilize
            time.sleep(1)

            while current_attempt < max_attempts:
                try:
                    frame = kinect.video_stream.get_next_frame()
                    if frame and frame.image:
                        break
                except WindowsError as win_err:
                    print("WindowsError:", win_err)
                    current_attempt += 1
                    time.sleep(0.2)  

            if not frame or not frame.image:
                print("Error: Failed to capture frame from Kinect.")
                return None

            color_data = np.empty((480, 640, 4), np.uint8)
            frame.image.copy_bits(color_data.ctypes.data)
            color_image = color_data[..., :3]

            if check_image_shape(color_image):
                img_name = "captured_image_%d.jpg" % attempt
                img_path = os.path.join(capture_output_dir, img_name)
                cv2.imwrite(img_path, color_image)
                return img_path
            else:
                print("Error: Captured image does not have expected shape or channels.")
                return None
        else:
            print("Kinect is not connected.")
            return None
    except Exception as e:
        print("Error capturing color image:", e)
        return None


def calculate_homography():
    global H
    H, _ = cv2.findHomography(pts1, pts2)
    nx, ny = (472, 612)
    x = np.linspace(1, 472, nx)
    y = np.linspace(1, 612, ny)
    xv, yv = np.meshgrid(x, y)
    np.savetxt('xv.txt', xv, fmt='%.2f')
    np.savetxt('yv.txt', yv, fmt='%.2f')
    m1 = xv.flatten()
    n1 = yv.flatten()
    w = np.array([m1, n1, np.ones(288864)])
    H_inv = np.linalg.inv(H)
    w1 = np.dot(H_inv, w)
    x_cor = w1[0]
    y_cor = w1[1]
    x_corr = x_cor.reshape(612, 472)
    x_corr = x_corr.transpose()
    y_corr = y_cor.reshape(612, 472)
    y_corr = y_corr.transpose()
    np.savetxt('x_corre.txt', x_corr, fmt='%.2f')
    np.savetxt('y_corre.txt', y_corr, fmt='%.2f')
    return x_corr, y_corr , xv,yv

x_corr, y_corr ,xv, yv= calculate_homography()

def interpolation():
    global x_corr, y_corr, xv, yv, interpolated_depth
    try:
        if is_kinect_connected():
            if global_depth_data is not None:
                interpolated_depth = griddata((x_corr.flatten(), y_corr.flatten()), global_depth_data.flatten(), (xv, yv), method='nearest')
                matrix_size = interpolated_depth.shape
                # print("Matrix size:", matrix_size)
            else:
                print('No global depth data.')
        else:
            status_label.config(text="Kinect unconnected", font=('Roboto',10))
        return interpolated_depth
    except Exception as e:
        print("Error:", e)

def update_plot():
    if not animation_running:
        return  # Stop the animation if the flag is set to False
    
    start_time = time.time()
    interpolated_depth = interpolation()
    duration = time.time() - start_time

    global interpolated_depth_plot

    if interpolated_depth is not None:
        if interpolated_depth_plot is None:
            fig = plt.figure(figsize=(1300 / 100, 800 / 100))
            fig.canvas.manager.window.wm_geometry("+1530+-100")
            fig.canvas.manager.window.overrideredirect(1)
            fig.canvas.manager.window.state('zoomed')
            hAx = plt.gca()
            hAx.set_position([0, 0, 1, 1])
            plt.axis('off')
            ax = fig.add_subplot(111)
            start_time2 = time.time()
            interpolated_depth_plot = ax.imshow(interpolated_depth, cmap='jet', aspect='auto', origin='lower', norm=LogNorm(vmin=interpolated_depth.min() + 1, vmax=interpolated_depth.max()))
            duration2 = time.time() - start_time2
            print("show", duration2)
        else:
            interpolated_depth_plot.set_data(interpolated_depth.transpose())
            interpolated_depth_plot.autoscale()
            plt.pause(0.00001)

    root.after(0, update_plot)
    print("inter", duration)

def start_animation():
    global animation_running
    animation_running = True
    update_plot()

def stop_animation():
    global animation_running
    animation_running = False

def subtract_matrix_from_depth(matrix_file, interpolated_depth):
    try:
        if matrix_file:
            with open(matrix_file, 'r') as file:
                # matrix = [[float(num) for num in line.split()] for line in file]
                matrix = [[float(num) for num in line.split(',')] for line in file]
            
            # Convert the matrix to a NumPy array
            matrix = np.array(matrix)
            matrix = np.transpose(matrix)
            if matrix.shape == (612, 472):
                if interpolated_depth is not None and interpolated_depth.shape == (612, 472):
                    subtracted_data = np.subtract(matrix, interpolated_depth)
                    return subtracted_data
                else:
                    print("Dimension mismatch between global depth data and matrix from file.")
            else:
                print("Matrix dimensions are not 612x472.")
        else:
            print("No matrix file selected.")

    except Exception as e:
        print("Error subtracting matrix from depth:", e)

# Example usage:
# subtracted_data = subtract_matrix_from_depth("matrix.txt", interpolated_depth)

def plot_mnt():
    global  interpolated_depth
    if not substraction_running:
        return
    start_time = time.time()
    interpolated_depth = interpolation()  # Assuming interpolation function is defined
    duration = time.time() - start_time
    global comparison, comparison_matrix, matrix, interpolated_depth_plot, sub_data, mnt

    if mnt is not None:

        if interpolated_depth is not None:
            sub_data = subtract_matrix_from_depth(mnt, interpolated_depth)

            #filtered_data = apply_FCD(sub_data)

            # Generate comparison matrix based on filtered data
            comparison_matrix = np.zeros_like(sub_data)
            comparison_matrix[(sub_data <= 10) & (sub_data >= -10)] = 0
            comparison_matrix[sub_data > 10] = -50
            comparison_matrix[sub_data < -10] = 50

            if sub_data is not None:
                if comparison is None:
                    fig = plt.figure(figsize=(1300 / 100, 800 / 100))
                    fig.canvas.manager.window.wm_geometry("+1530+-100")
                    fig.canvas.manager.window.overrideredirect(1)
                    fig.canvas.manager.window.state('zoomed')
                    hAx = plt.gca()
                    hAx.set_position([0, 0, 1, 1])
                    plt.axis('off')
                    ax = fig.add_subplot(111)
                    print("comparison", comparison_matrix)

                    comparison = ax.imshow(comparison_matrix, cmap="jet", aspect='auto',origin="lower",norm=SymLogNorm(linthresh=1, vmin=-50, vmax=50))

                    # Plot the data
                else:
                    comparison.set_data(comparison_matrix.transpose())
                    comparison.autoscale()
                    plt.pause(0.00001)
            root.after(0, plot_mnt())
            print("duration",duration)
    else:
        mnt = upload_file()
        plot_mnt()

def upload_file():
    global matrix_file
    try:
        file_types = [
            ("TXT Files", "*.txt"),
            ("TIFF Files", "*.tif"),
            ("All Files", "*.*")
        ]
        matrix_file = tkFileDialog.askopenfilename(filetypes=file_types)
        if matrix_file:
            print("File uploaded:", matrix_file)
            return matrix_file
        else:
            print("No file selected.")
    except Exception as e:
        print("Error uploading file:", e)

def start_substraction():
    global substraction_running
    substraction_running = True
    plot_mnt()

def stop_substraction():
    global substraction_running
    substraction_running = False

def upload_image():
    global image_file
    try:
        file_types = [
            ("Image Files", "*.jpg;*.jpeg;*.png;*.gif"),
            ("All Files", "*.*")
        ]
        image_file = tkFileDialog.askopenfilename(filetypes=file_types)
        if image_file:
            print("Image uploaded:", image_file)
            return image_file
        else:
            print("No image selected.")
    except Exception as e:
        print("Error uploading image:", e)

def sat_img():
    image = upload_image()
    fig = plt.figure(figsize=(1300 / 100, 800 / 100))
    fig.canvas.manager.window.wm_geometry("+1530+-100")
    fig.canvas.manager.window.overrideredirect(1)
    fig.canvas.manager.window.state('zoomed')
    hAx = plt.gca()
    hAx.set_position([0, 0, 1, 1])
    plt.axis('off')
    image = mpimg.imread(image)

    # Display the image
    plt.imshow(image)
    plt.axis('auto')  # Set the aspect ratio to auto

    # Store the axis object if needed
    ax = plt.gca()

    # Show the plot
    plt.show()

def calculate_viewshed(observer_point, FOV_degrees, dem_array):
    observer_x, observer_y, observer_z = observer_point
    dem_height, dem_width = dem_array.shape
    
    def check_line_of_sight(x, y):
        elevation = dem_array[y, x]
        dx = x - observer_x
        dy = y - observer_y
        distance = math.sqrt(dx**2 + dy**2)
        azimuth = math.atan2(dy, dx)
        if distance > 0:
            angle = math.atan((elevation - observer_z) / distance)
        else:
            angle = 0
        if angle <= math.radians(FOV_degrees/2):
            los_clear = True
            for i in range(1, int(distance)):
                tx = observer_x + i * math.cos(azimuth)
                ty = observer_y + i * math.sin(azimuth)
                televation = dem_array[int(ty), int(tx)]
                if televation > observer_z:
                    los_clear = False
                    break
            return 1 if los_clear else 0

    results = Parallel(n_jobs=-1)(delayed(check_line_of_sight)(x, y) for y in range(dem_height) for x in range(dem_width))
    viewshed_array = np.array(results).reshape((dem_height, dem_width))
    return viewshed_array

def plot_viewshed(viewshed_array, observer_point):
    x, y, _ = observer_point
    
    viewshed_array = np.clip(viewshed_array, 0, 1)
    fig = plt.figure(figsize=(1300 / 100, 800 / 100))
    fig.canvas.manager.window.wm_geometry("+1530+-100")
    fig.canvas.manager.window.overrideredirect(1)
    fig.canvas.manager.window.state('zoomed')
    hAx = plt.gca()
    hAx.set_position([0, 0, 1, 1])
    plt.axis('off')
    plt.imshow(viewshed_array, cmap='gray')
    plt.axis('auto')
    plt.scatter(x, y, marker='x', color='red', s=100)
    plt.show()

def calculate_and_plot_viewshed():
    global mnt
    observer_x = int(input("Enter observer x-coordinate: "))
    observer_y = int(input("Enter observer y-coordinate: "))
    observer_z = int(input("Enter observer z-coordinate: "))
    FOV_degrees = 10  # Field of view in degrees
    data = np.loadtxt(mnt)
    def array_to_raster(array, output_file):
        rows, cols = array.shape
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)
        out_raster.SetGeoTransform((0, 1, 0, 0, 0, -1))  # Placeholder geotransform
        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(array)
        out_band.SetNoDataValue(-9999)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # Assuming WGS84 lat/lon
        out_raster.SetProjection(srs.ExportToWkt())
        out_band.FlushCache()

    array_to_raster(data, 'MNT')

    dem_dataset = gdal.Open(dem_file)
    if dem_dataset is None:
        print("Error: DEM file not found or invalid.")
        return
    
    dem_array = dem_dataset.GetRasterBand(1).ReadAsArray()
    
    observer_point = (observer_x, observer_y, observer_z)
    viewshed_array = calculate_viewshed(observer_point, FOV_degrees, dem_array)
    plot_viewshed(viewshed_array, observer_point)
    
    # Close the GDAL dataset
    dem_dataset = None

def on_enter(event):
    event.widget.config(bg="#f5f6fa",fg='#3C4AF1')#DCDEEC

def on_leave(event):
    event.widget.config(bg='#3C4AF1',fg='white')
#------------------------------------------------------------------------------------------------------------#
# Interface 
#-----------

root = tk.Tk()
root.title("SandBox")
root.geometry("980x560")
root.configure(bg="#f5f6fa")
# root.iconbitmap(r'C:\Users\pc\Desktop\interface_projet\try2\icon.ico')

title_label = tk.Label(root, text="SANDBOX", font=("Arial", 16, "bold"), bg="#f5f6fa", fg='black')
title_label.pack(pady=16)

left_frame = tk.Frame(root, bd=2, width=200, height=450,bg='#f5f6fa') #bg="#DCDEEC")
left_frame.pack(side=tk.LEFT, pady=15, fill=tk.Y)

# Dimensions fixes pour tous les frames
frame_width = 300
frame_height = 100

button_font = ('Roboto',10)

# Configuration de button_frame
button_frame = tk.Frame(left_frame, bd=2, bg="#FDFDFF",   width=frame_width, height=frame_height)
button_frame.pack( pady=7, padx=20)

status_label = tk.Label(button_frame, text="", bg="#FDFDFF", fg='black')
status_label.grid(row=0, column=1, padx=10, pady=5)

connect_button = tk.Button(button_frame, width=15, height=2, bg="#3C4AF1", fg="white", text='Connect', font=button_font, command=connect_kinect)
connect_button.grid(row=0, column=0, padx=10, pady=5)
connect_button.bind("<Enter>", on_enter)
connect_button.bind("<Leave>", on_leave)

calibrate_button = tk.Button(button_frame, width=15, height=2, bg="#3C4AF1", fg="white", text='Calibrate', font=button_font, command=lambda: project_checkerboard_images_and_capture(generated_checkerboard_dir, capture_output_dir, kinect))
calibrate_button.grid(row=1, column=0, padx=10, pady=5)
calibrate_button.bind("<Enter>", on_enter)
calibrate_button.bind("<Leave>", on_leave)

plot_button = tk.Button(button_frame, width=15, height=2, bg="#3C4AF1", fg="white", text='Plot', font=button_font, command= display_depth)
plot_button.grid(row=1, column=1, padx=10, pady=5)
plot_button.bind("<Enter>", on_enter)
plot_button.bind("<Leave>", on_leave)

# Configuration de button_frame_2
button_frame_2 = tk.Frame(left_frame, bd=2, bg="#FDFDFF",   width=frame_width, height=frame_height)
button_frame_2.pack( pady=7, padx=20)

calibration_label = tk.Label(button_frame_2, text="INTERPOLATION", font=("Arial", 10, 'bold'), bg="#FDFDFF", fg='black')
calibration_label.grid(row=0, column=0, columnspan=2, padx=10)

START_button = tk.Button(button_frame_2, width=15, height=2, bg="#3C4AF1", fg="white", text='Start', font=button_font, command=start_animation)
START_button.grid(row=1, column=0, padx=10, pady=5)
START_button.bind("<Enter>", on_enter)
START_button.bind("<Leave>", on_leave)

STOP_button = tk.Button(button_frame_2, width=15, height=2, bg="#3C4AF1", fg="white", text='Stop', font=button_font, command=stop_animation)
STOP_button.grid(row=1, column=1, padx=10, pady=5)
STOP_button.bind("<Enter>", on_enter)
STOP_button.bind("<Leave>", on_leave)

# Configuration de button_frame_3
button_frame_3 = tk.Frame(left_frame, bd=2, bg="#FDFDFF",   width=frame_width, height=frame_height)
button_frame_3.pack(padx=20, pady=7)

mnt_label = tk.Label(button_frame_3, text="DEM", font=("Arial", 10, 'bold'), bg="#FDFDFF", fg='black')
mnt_label.grid(row=0, column=0, columnspan=2, padx=10)

start_button_2 = tk.Button(button_frame_3, width=15, height=2, bg="#3C4AF8", fg="white", text='Start', font=button_font, command=start_substraction)
start_button_2.grid(row=1, column=0, padx=10, pady=5)
start_button_2.bind("<Enter>", on_enter)
start_button_2.bind("<Leave>", on_leave)

stop_button_2 = tk.Button(button_frame_3, width=15, height=2, bg="#3C4AF1", fg="white", text='Stop', font=button_font, command=stop_substraction)
stop_button_2.grid(row=1, column=1, padx=10, pady=5)
stop_button_2.bind("<Enter>", on_enter)
stop_button_2.bind("<Leave>", on_leave)

satili_button = tk.Button(button_frame_3, width=25, height=2, bg="#3C4AF1", fg="white", text='Satellite Image', font=button_font, command= sat_img)
satili_button.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
satili_button.bind("<Enter>", on_enter)
satili_button.bind("<Leave>", on_leave)

# Configuration de button_frame_4
button_frame_4 = tk.Frame(left_frame, bd=2, bg="#FDFDFF",   width=frame_width, height=frame_height+10)
button_frame_4.pack(padx=20, pady=7)

vision_label = tk.Label(button_frame_4, text="FIELD OF VIEW", font=("Arial", 10, 'bold'), bg="#FDFDFF", fg='black')
vision_label.grid(row=0, column=0, columnspan=2, padx=10)

# label_x = Label(button_frame_4, text="X:")
# label_x.grid(row=0, column=0)
# entry_x = tk.Entry(button_frame_4)
# entry_x.grid(row=0, column=1)

# label_y = Label(button_frame_4, text="Y:")
# label_y.grid(row=1, column=0)
# entry_y = tk.Entry(button_frame_4)
# entry_y.grid(row=1, column=1)

visualiser_button = tk.Button(button_frame_4, width=25, height=2, bg="#3C4AF1", fg="white", text='Visualize', font=button_font) #, command=calculate_and_plot_viewshed)
visualiser_button.grid(row=2, column=0, columnspan=2, padx=10, pady=3)
visualiser_button.bind("<Enter>", on_enter)
visualiser_button.bind("<Leave>", on_leave)

# Create rigth_frame
right_frame = tk.Frame(root, bd=2, width=500, height=600 , bg="#f5f6fa")
right_frame.pack(side=tk.RIGHT, padx=10, pady=22, fill=tk.Y)

# create image_frame for ploting
image_frame = tk.Frame(right_frame, width=550, height=420, bd=2, relief=tk.SUNKEN)
image_frame.grid(row=0, column=0, padx=10)

# Ajout d'un widget Canvas pour afficher l'image dans le cadre d'image
canvas = tk.Canvas(image_frame, width=550, height=420, bg='#f5f9fa')#bg="#c4ccc2")
canvas.pack()

root.mainloop()