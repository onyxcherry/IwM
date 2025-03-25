from math import pi, sqrt, floor, ceil
import os

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.draw import line_nd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import perf_counter
from line_profiler import LineProfiler


def load_bitmap(filename: str) -> ImageFile:
    image = Image.open("./obrazy/" + filename)
    return image


def find_coords_inside_image(
    pixels: NDArray,
    im_size: tuple,
) -> tuple:
    # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
    width = im_size[0]
    height = im_size[1]

    line_start = 0
    line_end = pixels.shape[1] - 1
    mid_line_point = (line_start + line_end) // 2

    pixel_x, pixel_y = pixels[:, mid_line_point]
    while not (0 <= pixel_x < width and 0 <= pixel_y < height):
        return (-1, -1)

    end = mid_line_point
    # print(f"1. {line_start} : {end}")
    while line_start != end:
        mid_point = (line_start + end) // 2
        pixel_x, pixel_y = pixels[:, mid_point]
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            end = mid_point
        else:
            line_start = mid_point + 1
        # print(f"1. {line_start} : {end}")

    start = mid_line_point
    # print(f"2. {start} : {line_end}")
    while start != line_end:
        mid_point = ((start + line_end) // 2) + 1
        pixel_x, pixel_y = pixels[:, mid_point]
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            start = mid_point
        else:
            line_end = mid_point - 1

    return (line_start, line_end)


def line_intersection(A, B, C, D):
    # a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    # determinant
    det = a1 * b2 - a2 * b1

    # parallel line
    if det == 0:
        return (None, None)

    # intersect point(x,y)
    x = ((b2 * c1) - (b1 * c2)) / det
    y = ((a1 * c2) - (a2 * c1)) / det

    min_x = min(C[0], D[0])
    max_x = max(C[0], D[0])
    min_y = min(C[1], D[1])
    max_y = max(C[1], D[1])

    isInsideSegments = (min_x <= x and 
                        x <= max_x and
                        min_y <= y and
                        y <= max_y)
    
    return (x, y) if isInsideSegments else (None, None)



def line_image_intersection(start, coords, masks, im_size):
    points = coords[masks]

    # max_x = im_size[0] - 1 
    # max_y = im_size[1] - 1 

    # if masks[0]:
    #     points.append([0, coords[0][1]])
    
    # if masks[1]:
    #     points.append([max_x, coords[1][1]])

    # if masks[2]:
    #     points.append([coords[2][0], 0])

    # if masks[3]:
    #     points.append([coords[3][0], max_y])



    if points.shape[0] == 0:
        return (None, None)

    if len(points) == 1:
        return (points[0], points[0])

    return points

    
def get_line_pixels(start, end):
    # d0, d1 = np.diff([start,end], axis=0)[0]
    if start[0] == end[0] and start[1] == end[1]:
        return (np.array([int(start[0])]), np.array([int(start[1])]))
    d0 = end[0] - start[0] 
    d1 = end[1] - start[1] 
    if np.abs(d0) > np.abs(d1):
        x = np.arange(start[0], end[0] + np.sign(d0), np.sign(d0), dtype=np.int32)
        n = len(x)
        y = np.linspace(start[1], end[1], n, endpoint=True, dtype=np.int32)
        # step = d1/(n-1)
        # y = np.arange(start[1], end[1] + step, step, dtype=np.int32)
        return (x,y)
    else:
        y = np.arange(start[1], end[1] + np.sign(d1), np.sign(d1), dtype=np.int32)
        n = len(y)
        x = np.linspace(start[0], end[0], n, endpoint=True, dtype=np.int32)
        # step = d0/(n-1)
        # x = np.arange(start[0], end[0] + step, step, dtype=np.int32)
        return (x,y)

def get_line_pixels1(a, im_size, max_size: int):
    # d0, d1 = np.diff([start,end], axis=0)[0]
    matrix = np.full((max_size, max_size), False)
    d0 = a[2] - a[0] 
    d1 = a[3] - a[1] 
    if np.abs(d0) > np.abs(d1):
        x = np.arange(a[0], a[2] + np.sign(d0), np.sign(d0), dtype=np.int32)
        n = len(x)
        y = np.linspace(a[1], a[3], n, endpoint=True, dtype=np.int32)
        # step = d1/(n-1)
        # y = np.arange(a[1], a[3] + step, step, dtype=np.int32)
        matrix[(x,y)] = True
        return np.packbits(matrix[:im_size[0], :im_size[1]])
        # return (x,y)
    else:
        y = np.arange(a[1], a[3] + np.sign(d1), np.sign(d1), dtype=np.int32)
        n = len(y)
        x = np.linspace(a[0], a[2], n, endpoint=True, dtype=np.int32)
        # step = d0/(n-1)
        # x = np.arange(a[0], a[2] + step, step, dtype=np.int32)
        matrix[(x,y)] = True
        return np.packbits(matrix[:im_size[0], :im_size[1]])
        # return (x,y)

def calculate_scan_rays(
    image_filename: str,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
    folder_name: str = "",
    isTest: bool = False,
    create_optional_partial_images: bool = False,
    create_optional_full_image: bool = False,
) -> NDArray:
# 1.3% time
    image = load_bitmap(image_filename)
    image.load()
    im_size = image.size
    image.close()

    # 1. Opisz okrąg na obrazie
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2

    im_left_x = -im_size[0] // 2
    im_top_y = -im_size[1] // 2

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu

    # Arrays of all emitters and detectors positions set to (radius, radius)
    emitters_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )
    detectors_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )

    # Arrays of all emitters angles set to 0 rad
    emitters_angles = np.zeros((scans_count, detectors_count), dtype=np.float64)

    # Arrays of all detectors angles set to pi rad
    detectors_angles = np.full((scans_count, detectors_count), pi, dtype=np.float64)

    # Array of angle differences between scans
    scan_start_angles = np.arange(
        0, scans_count * alfa_step, alfa_step, dtype=np.float64
    )

    # Update emitters and detectors angles to include difference in angle between scans
    emitters_angles += scan_start_angles[:, np.newaxis]
    detectors_angles += scan_start_angles[:, np.newaxis]

    # Array of angle differences between detectors/emitters
    angles_dif = np.linspace(
        -detectors_angular_aperture / 2,
        detectors_angular_aperture / 2,
        detectors_count,
        dtype=np.float64,
    )

    # Update emitters and detectors angles to include difference in angle between detectors/emitters
    emitters_angles -= angles_dif
    detectors_angles += angles_dif

    # Calculate cordinates of emitters and detectors
    emitters_positions_x = np.cos(emitters_angles)
    emitters_positions_y = np.sin(emitters_angles)
    emitters_positions_xy = np.stack(
        (emitters_positions_x, emitters_positions_y), axis=2, dtype=np.float64
    )
    emitters_positions *= emitters_positions_xy

    detectors_positions_x = np.cos(detectors_angles)
    detectors_positions_y = np.sin(detectors_angles)
    detectors_positions_xy = np.stack(
        (detectors_positions_x, detectors_positions_y), axis=2, dtype=np.float64
    )
    detectors_positions_xy = np.stack(
        (np.cos(detectors_angles), np.sin(detectors_angles)), axis=2, dtype=np.float64
    )
    detectors_positions *= detectors_positions_xy

    emitters_positions -= np.array([im_left_x, im_top_y])[np.newaxis, np.newaxis, :]
    detectors_positions -= np.array([im_left_x, im_top_y])[np.newaxis, np.newaxis, :]

    if isTest:
        star_end_points = np.concatenate([emitters_positions, detectors_positions], axis=-1)
        return np.apply_along_axis(get_line_pixels1, axis=-1, arr=star_end_points, im_size=im_size, max_size=int(diagonal_length))


    max_x = im_size[0] - 1 
    max_y = im_size[1] - 1 

    
    a_array = detectors_positions[:,:,1] - emitters_positions[:,:,1]
    b_array = emitters_positions[:,:,0] - detectors_positions[:,:,0]
    c_array = a_array * emitters_positions[:,:,0] + b_array * emitters_positions[:,:,1]

    # print(a_array.shape)
    # print(b_array.shape)
    # print(c_array.shape)

    y1 = np.divide(c_array, b_array, out=np.full_like(c_array, -1), where=b_array!=0)
    y2 = np.divide(c_array - (a_array * max_x), b_array, out=np.full_like(c_array, -1), where=b_array!=0)
    x1 = np.divide(c_array, a_array, out=np.full_like(c_array, -1), where=a_array!=0)
    x2 = np.divide(c_array - (b_array * max_y), a_array, out=np.full_like(c_array, -1), where=a_array!=0)
    # print("----------------------------------------------")
    # print(y1.shape)
    # print(y2.shape)
    # print(x1.shape)
    # print(x2.shape)
    coords1 = np.stack([np.zeros_like(y1),y1], axis=-1)
    coords2 = np.stack([np.full_like(y2, max_x),y2], axis=-1)
    coords3 = np.stack([x1,np.zeros_like(x1)], axis=-1)
    coords4 = np.stack([x2,np.full_like(x2, max_y)], axis=-1)
    coords = np.stack([coords1, coords2, coords3, coords4], axis=-2)
    masks1 = ((0 <= coords1[:,:,1]) & (coords1[:,:,1] <= max_y))
    masks2 = ((0 <= coords2[:,:,1]) & (coords2[:,:,1] <= max_y))
    masks3 = ((0 < coords3[:,:,0]) & (coords3[:,:,0] < max_x))
    masks4 = ((0 < coords4[:,:,0]) & (coords4[:,:,0] < max_x))
    masks = np.stack([masks1, masks2, masks3, masks4], axis=-1)

    # print("----------------------------------------------")
    # print(emitters_positions.shape)
    # print(detectors_positions.shape)
    # print(star_end_points.shape)
    # print(pixels.shape)
    # print(np.unpackbits(pixels, axis=2, count=im_size[0]*im_size[1]).reshape((scans_count, detectors_count, im_size[0], im_size[1])).shape)
    # print("----------------------------------------------")
    # print(emitters_positions[0][2])
    # print(detectors_positions[0][2])
    # print(star_end_points[0][2])
    # print(pixels[0][2])
    # print(np.unpackbits(pixels, axis=2, count=im_size[0]*im_size[1]).reshape((scans_count, detectors_count, im_size[0], im_size[1]))[0][2])
    # print(np.unpackbits(pixels, axis=2, count=im_size[0]*im_size[1]).reshape((scans_count, detectors_count, im_size[0], im_size[1]))[0][2].sum())


    # pixels_coords1 = np.apply_along_axis()
    # points =  np.stack([coords1[masks1], coords2[masks2], coords3[masks3], coords4[masks4]], axis=-2)

    # if 0 <= coords[0][1] <= max_y:
    #     points.append([0, coords[0][1]])
    
    # if 0 <= coords[1][1] <= max_y:
    #     points.append([max_x, coords[1][1]])

    # if 0 < coords[2][0] < max_x:
    #     points.append([coords[2][0], 0])

    # if 0 < coords[3][0] < max_x:
    #     points.append([coords[3][0], max_y])

    # print("----------------------------------------------")
    # print(coords1.shape)
    # print(coords2.shape)
    # print(coords3.shape)
    # print(coords4.shape)
    # print("----------------------------------------------")
    # print(coords.shape)
    # print(masks.shape)
    # print(points.shape)
    # print("----------------------------------------------")
    # print(masks.min())
    # print(masks.max())
    # print(coords.sum())

    all_scans_rays = np.empty((scans_count, detectors_count), dtype=np.object_)
    for scan_number in range(scans_count):
        for detector_number in range(detectors_count):
            # 3. Wyznacz linie przejścia
            # 3. a) wyznacz linie przejścia od emiterów do detektorów
            emitter_coords = emitters_positions[scan_number][detector_number]
            detector_coords = detectors_positions[scan_number][detector_number]

            start, end = line_image_intersection(emitter_coords, coords[scan_number, detector_number], masks[scan_number, detector_number], im_size)
            # start, end = line_image_intersection(emitter_coords, detector_coords, im_size)
            if start is None:
                all_scans_rays[scan_number][detector_number] = tuple(([], []))
                continue

            pixels_coords = get_line_pixels(start, end)
            all_scans_rays[scan_number][detector_number] = pixels_coords
            continue

# 47.3% time
            # pixels_coords = line_nd(emitter_coords, detector_coords, endpoint=True)
            pixels_coords = get_line_pixels(emitter_coords, detector_coords)

            # Change coordinate system to image coordinate system where point (0,0) means upper left corner of image
# 6.2% time
            pixels_coords -= np.array([im_left_x, im_top_y])[:, np.newaxis]

            # 3. b) wyznacz linie przejścia tylko przez obraz (wersja z wyszukiwaniem przez połowienie)
# 15.7% time
            line_start, line_end = find_coords_inside_image(pixels_coords, im_size)
            if line_start == -1:
                # Alternative way to calculate similiar time requirements
                width = im_size[0]
                height = im_size[1]
                filter = (
                    (0 <= pixels_coords[0, :])
                    & (pixels_coords[0, :] < width)
                    & (0 <= pixels_coords[1, :])
                    & (pixels_coords[1, :] < height)
                )

                all_scans_rays[scan_number][detector_number] = tuple(
                    pixels_coords[:, filter]
                )
            else:
                all_scans_rays[scan_number][detector_number] = tuple(
                    pixels_coords[:, line_start : line_end + 1]
                )
            

        # Optional partial rays image
        if create_optional_partial_images:
            if folder_name == "":
                folder_name = image_filename.split(".")[0]
            plt.figure(figsize=(8, 8))
            plt.axis("equal")
            plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
            plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
            x, y = zip(*(emitters_positions[scan_number]))
            plt.plot(x, y, "ro", markersize=2)
            x, y = zip(*(detectors_positions[scan_number]))
            plt.plot(x, y, "go", markersize=2)
            all_ray_pixels_x = []
            all_ray_pixels_y = []
            for ray in all_scans_rays[scan_number]:
                x, y = ray
                all_ray_pixels_x.extend(np.array(x) + im_left_x)
                all_ray_pixels_y.extend(np.array(y) + im_top_y)
            x, y = all_ray_pixels_x, all_ray_pixels_y
            plt.plot(x, y, "bo", markersize=1, alpha=0.1)
            plt.imshow(
                plt.imread(f"obrazy/{image_filename}"),
                extent=(
                    im_left_x,
                    im_left_x + im_size[0],
                    im_top_y + im_size[1],
                    im_top_y,
                ),
            )
            if not os.path.exists(f"{folder_name}/rays"):
                if not os.path.exists(f"{folder_name}"):
                    os.mkdir(f"{folder_name}")
                os.mkdir(f"{folder_name}/rays")
            plt.savefig(f"{folder_name}/rays/pixels_{scan_number}.png")

    # Optional full rays image
    if create_optional_full_image:
        if folder_name == "":
            folder_name = image_filename.split(".")[0]
        diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
        radius = diagonal_length / 2
        plt.figure(figsize=(8, 8))
        plt.axis("equal")
        plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
        plt.ylim((ceil(radius * 1.1), -floor(radius * 1.1)))
        all_emitters_pixels_x = []
        all_emitters_pixels_y = []
        for scan_emitters_positions in emitters_positions:
            for emitter_position in scan_emitters_positions:
                x, y = emitter_position
                all_emitters_pixels_x.append(x)
                all_emitters_pixels_y.append(y)
        plt.plot(all_emitters_pixels_x, all_emitters_pixels_y, "go", markersize=2)
        plt.plot(x, y, "ro", markersize=2)
        all_detectors_pixels_x = []
        all_detectors_pixels_y = []
        for scan_detectors_positions in detectors_positions:
            for detector_position in scan_detectors_positions:
                x, y = detector_position
                all_detectors_pixels_x.append(x)
                all_detectors_pixels_y.append(y)
        plt.plot(all_detectors_pixels_x, all_detectors_pixels_y, "go", markersize=2)
        all_ray_pixels_x = []
        all_ray_pixels_y = []
        for scan_rays in all_scans_rays:
            for ray in scan_rays:
                x, y = ray
                all_ray_pixels_x.extend(np.array(x) + im_left_x)
                all_ray_pixels_y.extend(np.array(y) + im_top_y)
        plt.plot(all_ray_pixels_x, all_ray_pixels_y, "bo", markersize=1, alpha=0.1)
        plt.imshow(
            plt.imread(f"obrazy/{image_filename}"),
            extent=(
                im_left_x,
                im_left_x + im_size[0],
                im_top_y + im_size[1],
                im_top_y,
            ),
        )
        if not os.path.exists(f"{folder_name}/rays"):
            if not os.path.exists(f"{folder_name}"):
                os.mkdir(f"{folder_name}")
            os.mkdir(f"{folder_name}/rays")
        plt.savefig(f"{folder_name}/rays/pixels.png")

    return all_scans_rays


def calculate_sinogram(
    image_filename: str,
    all_scans_rays: NDArray,
    isTest: bool = False
) -> NDArray:
    # Create array for sinogram data

    image = load_bitmap(image_filename)
    image.load()
    im_size = image.size
    im_pixels = np.asarray(image, dtype=np.float64)
    image.close()
    im_pixels1 = im_pixels.mean(axis=2)

    scans_count, detectors_count = all_scans_rays.shape[:2]
    data = np.zeros((scans_count, detectors_count), dtype=np.float64)

    if isTest:
        masks = np.unpackbits(all_scans_rays, axis=-1, count=im_size[0]*im_size[1]).reshape((all_scans_rays.shape[0], all_scans_rays.shape[1], im_size[0], im_size[1])).view(np.bool)
        data = np.mean(np.broadcast_to(im_pixels1[np.newaxis,np.newaxis, :, :]
                                       , (all_scans_rays.shape[0], all_scans_rays.shape[1], im_size[0], im_size[1]))
                       , (-2,-1)
                       , out=np.zeros((all_scans_rays.shape[0], all_scans_rays.shape[1]))
                       , where=masks)
        data[data!=data] = 0
        return data
            



    for scan_number in range(scans_count):
        for detector_number in range(detectors_count):
            if len(all_scans_rays[scan_number, detector_number][0]) == 0:
                data[scan_number, detector_number] = 0
                continue
# 3.6%
            # print(im_pixels1[all_scans_rays[scan_number, detector_number]])

            data[scan_number, detector_number] = np.mean(im_pixels1[all_scans_rays[scan_number, detector_number]])
            # print(data[scan_number, detector_number])

#             ray_matrix = np.zeros(im_size, dtype=np.uint8)
# # 4.1%
#             ray_matrix[all_scans_rays[scan_number, detector_number]] = 1

# # 57.5% time
#             ray_pixel_number = ray_matrix.sum()

# # 26.1% time
#             ray_sum = im_pixels1.sum(where=(ray_matrix > 0))

#             if ray_pixel_number == 0:
#                 print(all_scans_rays[scan_number, detector_number])
#                 print(len(all_scans_rays[scan_number, detector_number]))
#                 print(ray_matrix)
#                 print(ray_sum)
#                 exit()


#             data[scan_number, detector_number] = ray_sum / ray_pixel_number

    # 5. Zwróć sinogram (bez filtracji)
    return data


def save_matrix_to_grayscale_image(matrix: NDArray, output_filename: str, has_to_be_scaled: bool = True):

    height, width = matrix.shape

    if has_to_be_scaled:
        matrix = (
            (matrix - np.min(matrix))
            / (np.max(matrix) - np.min(matrix))
            * (255)
        )

    image = Image.new("L", (width, height))  # "L" mode for grayscale
    image.putdata(matrix.flatten())
    image.save(output_filename)


def make_image(
    filename: str,
    all_scans_rays: NDArray,
    sinogram: NDArray,
    folder_name: str,
    isTest: bool = False
):
    scans_count, detectors_count = sinogram.shape

    image = load_bitmap(filename)
    im_size = image.size
    image.close()

    image_data = np.zeros(im_size)

    if isTest:
        masks = np.unpackbits(all_scans_rays, axis=-1, count=im_size[0]*im_size[1]).reshape((all_scans_rays.shape[0], all_scans_rays.shape[1], im_size[0], im_size[1])).view(np.bool)

    for scan_number in range(scans_count):
# 11.1% time
        for detector_number in range(detectors_count):
# 12.2% time            
            if isTest:
                image_data[masks[scan_number, detector_number]] += sinogram[scan_number, detector_number]
                continue

            if len(all_scans_rays[scan_number, detector_number][0]) == 0:
                continue

# 56.8% time
            image_data[all_scans_rays[scan_number, detector_number]] += sinogram[scan_number, detector_number]


    if not os.path.exists(f"{folder_name}"):
        os.mkdir(f"{folder_name}")
# 17.6% time
    save_matrix_to_grayscale_image(image_data, f"{folder_name}/obraz.png", False)


def filter_sinogram(sinogram: NDArray) -> NDArray:
    kernel = np.zeros(21)
    kernel[10] = 1
    numerator = -4/(pi**2)
    for k in range(1,11,2):
        kernel_value = (numerator)/(k**2)
        kernel[k+10] = kernel_value
        kernel[-k+10] = kernel_value

    filtered_sinogram = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"),axis=1, arr=sinogram)
    
    save_matrix_to_grayscale_image(filtered_sinogram, f"filtered_sinogram.png")

    return filtered_sinogram


def main():
    # filename = "Kropka.jpg"
    filename = "Shepp_logan.jpg"
    folder_name = ""
    scans_count = 720
    # scans_count = 80
    alfa_step = pi / scans_count
    detectors_count = 180
    # detectors_count = 80
    detectors_angular_aperture = 0.5 * pi

    if folder_name == "":
        folder_name = filename.split(".")[0]

    if not os.path.exists(f"{folder_name}"):
        os.mkdir(f"{folder_name}")

# 38% time
    scan_rays = calculate_scan_rays(
        filename, scans_count, alfa_step, detectors_count, detectors_angular_aperture, folder_name
    )
# 58.9% time
    sinogram = calculate_sinogram(filename, scan_rays)


    save_matrix_to_grayscale_image(sinogram, f"{folder_name}/sinogram.png")

    filtered_sinogram = filter_sinogram(sinogram)

    save_matrix_to_grayscale_image(filtered_sinogram, f"{folder_name}/filtered_sinogram.png")

# 2.8% time 
    make_image(filename, scan_rays, filtered_sinogram, folder_name)

    
    # scan_rays = calculate_scan_rays(
    #     filename, scans_count, alfa_step, detectors_count, detectors_angular_aperture, folder_name, True
    # )
    # sinogram = calculate_sinogram(filename, scan_rays, True)
    # save_matrix_to_grayscale_image(sinogram, f"{folder_name}/sinogram.png")
    # filtered_sinogram = filter_sinogram(sinogram)
    # save_matrix_to_grayscale_image(filtered_sinogram, f"{folder_name}/filtered_sinogram.png")
    # make_image(filename, scan_rays, filtered_sinogram, folder_name, True)


# Aplikacja powinna móc pozwolić konfigurować następujące elementy:
# Krok ∆α układu emiter/detektor.
# Dla jednego układu emiter/detektor liczbę detektorów (n).
# Rozwartość/rozpiętość układu emiter/detektor (l).


# 40x40
# 1 - normal sinogram without prescaling
# 2 - normal sinogram with prescaling
# 3 - filtered sinogram without prescaling
# 4 - filtered sinogram with prescaling
# 5 - filtered sinogram without image scaling
# 6 - filtered sinogram with prescaling without image scaling

# 80x160
# 11 - normal sinogram without prescaling
# 12 - normal sinogram with prescaling
# 13 - filtered sinogram without prescaling
# 14 - filtered sinogram with prescaling
# 15 - filtered sinogram without image scaling                  BEST 
# 16 - filtered sinogram with prescaling without image scaling

if __name__ == "__main__":
    lp = LineProfiler()
    # lp.add_function(line_image_intersection) 
    # lp.add_function(get_line_pixels) 
    # lp.add_function(calculate_scan_rays) 
    lp.add_function(calculate_sinogram)
    # lp.add_function(filter_sinogram)
    # lp.add_function(make_image)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
    # main()
