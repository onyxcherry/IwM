from dataclasses import dataclass
from math import pi, sqrt, floor, ceil
from typing import Optional

from PIL import Image
from PIL.ImageFile import ImageFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from IPython.display import clear_output

import pydicom
import datetime
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian


def load_bitmap(filename: str) -> ImageFile:
    image = Image.open("./obrazy/" + filename)
    return image


def calculateEmittersDetectorsPositions(
    image_start,
    radius,
    scans_count,
    detectors_count,
    alfa_step,
    detectors_angular_aperture,
):
    # All emitters and detectors positions set to (radius, radius)
    emitter_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )
    detector_positions = np.full(
        (scans_count, detectors_count, 2), radius, dtype=np.float64
    )

    # All emitters angles set to 0 rad
    emitter_angles = np.zeros((scans_count, detectors_count), dtype=np.float64)

    # All detectors angles set to pi rad
    detector_angles = np.full((scans_count, detectors_count), pi, dtype=np.float64)

    # Angle differences between scans
    scan_start_angles = np.arange(
        0, scans_count * alfa_step, alfa_step, dtype=np.float64
    )

    # Update emitters and detectors angles to include difference in angle between scans
    emitter_angles += scan_start_angles[:, np.newaxis]
    detector_angles += scan_start_angles[:, np.newaxis]

    # Angle differences between detectors/emitters
    detector_angle_differences = np.linspace(
        -detectors_angular_aperture / 2,
        detectors_angular_aperture / 2,
        detectors_count,
        dtype=np.float64,
    )

    # Update emitters and detectors angles to include difference in angle between detectors/emitters
    emitter_angles -= detector_angle_differences
    detector_angles += detector_angle_differences

    # Calculate coordinates of emitters
    emitter_positions_x = np.cos(emitter_angles, dtype=np.float64)
    emitter_positions_y = np.sin(emitter_angles, dtype=np.float64)
    emitter_positions_xy = np.stack(
        (emitter_positions_x, emitter_positions_y), axis=-1, dtype=np.float64
    )
    emitter_positions *= emitter_positions_xy

    # Calculate coordinates of detectors
    detector_positions_x = np.cos(detector_angles, dtype=np.float64)
    detector_positions_y = np.sin(detector_angles, dtype=np.float64)
    detector_positions_xy = np.stack(
        (detector_positions_x, detector_positions_y), axis=-1, dtype=np.float64
    )
    detector_positions *= detector_positions_xy

    # Shift coordinates to match with image
    emitter_positions -= image_start[np.newaxis, np.newaxis, :]
    detector_positions -= image_start[np.newaxis, np.newaxis, :]

    return emitter_positions, detector_positions


def calculateEmittersDetectorsPositionsEquiDistant(
    image_start,
    radius,
    scans_count,
    detectors_count,
    alfa_step,
    detectors_angular_aperture,
):
    # This method is only applicable for detectors_angular_aperture <= pi rad
    if detectors_angular_aperture > pi:
        return calculateEmittersDetectorsPositions(
            image_start,
            radius,
            scans_count,
            detectors_count,
            alfa_step,
            detectors_angular_aperture,
        )

    # Angle differences between scans
    scan_start_angles = np.arange(
        0, scans_count * alfa_step, alfa_step, dtype=np.float64
    )
    # Calculate values of sin and cos functions used to rotate final positions
    sin_array = np.sin(scan_start_angles, dtype=np.float64)
    cos_array = np.cos(scan_start_angles, dtype=np.float64)

    # Calculate scalars used for equidistant alligning rays
    detectors_position_span = (1 - np.cos(detectors_angular_aperture)) * radius
    detectors_position_diff = detectors_position_span / (detectors_count - 1)
    radius_2 = radius * radius

    # Calculations done for perfectly horizontal rays
    # Calculate points on a circle
    emitter_positions_y = np.arange(
        -detectors_position_span / 2,
        detectors_count * detectors_position_diff - detectors_position_span / 2,
        detectors_position_diff,
        dtype=np.float64,
    )
    emitter_positions_x = np.nan_to_num(
        np.sqrt(
            radius_2 - (emitter_positions_y * emitter_positions_y), dtype=np.float64
        )
    )
    detector_positions_y = emitter_positions_y.copy()
    detector_positions_x = -emitter_positions_x.copy()

    # Brodcast points to consider different positions on different scans
    emitter_positions_y = np.broadcast_to(
        emitter_positions_y[None, :], (scans_count, detectors_count)
    )
    emitter_positions_x = np.broadcast_to(
        emitter_positions_x[None, :], (scans_count, detectors_count)
    )
    detector_positions_y = np.broadcast_to(
        detector_positions_y[None, :], (scans_count, detectors_count)
    )
    detector_positions_x = np.broadcast_to(
        detector_positions_x[None, :], (scans_count, detectors_count)
    )

    # Rotate points by an angle related to scan number
    emitter_positions = np.stack(
        [
            emitter_positions_x * cos_array[:, None]
            + emitter_positions_y * sin_array[:, None],
            emitter_positions_y * cos_array[:, None]
            - emitter_positions_x * sin_array[:, None],
        ],
        axis=-1,
        dtype=np.float32,
    )
    detector_positions = np.stack(
        [
            detector_positions_x * cos_array[:, None]
            + detector_positions_y * sin_array[:, None],
            detector_positions_y * cos_array[:, None]
            - detector_positions_x * sin_array[:, None],
        ],
        axis=-1,
        dtype=np.float32,
    )

    # Shift positions in such way that they overlap with image pixels ( point (0,0) means pixel(0,0) )
    emitter_positions -= image_start[np.newaxis, np.newaxis, :]
    detector_positions -= image_start[np.newaxis, np.newaxis, :]

    return emitter_positions, detector_positions


def calcualateIntesections(
    image_width, image_height, emitter_positions, detector_positions
):
    # Calculate last indexes of image
    max_x = image_width - 1
    max_y = image_height - 1

    # Calculate standard forms of linear equation for all rays between detectors and emitters ( Ax + By = C )
    A_values = detector_positions[:, :, 1] - emitter_positions[:, :, 1]
    B_values = emitter_positions[:, :, 0] - detector_positions[:, :, 0]
    C_values = (
        A_values * emitter_positions[:, :, 0] + B_values * emitter_positions[:, :, 1]
    )

    # Calculate intersections of rays and image boundaries or -1 if lines are parallel
    # Round is for treating very small float numbers as 0 (1e-10 -> 0)
    # for x = 0 (left side of image)
    y1 = np.divide(
        C_values, B_values, out=np.full_like(C_values, -1), where=B_values != 0
    ).round(9)
    # for x = max_x (right side of image)
    y2 = np.divide(
        C_values - (A_values * max_x),
        B_values,
        out=np.full_like(C_values, -1),
        where=B_values != 0,
    ).round(9)
    # for y = 0 (top side of image)
    x1 = np.divide(
        C_values, A_values, out=np.full_like(C_values, -1), where=A_values != 0
    ).round(9)
    # remove points that are repeated
    x1[(x1 == 0) & (y1 == 0)] = -1
    x1[(x1 == max_x) & (y2 == 0)] = -1
    # for y = max_y (bottom side of image)
    x2 = np.divide(
        C_values - (B_values * max_y),
        A_values,
        out=np.full_like(C_values, -1),
        where=A_values != 0,
    ).round(9)
    # remove points that are repeated
    x2[(x2 == 0) & (y1 == max_y)] = -1
    x2[(x2 == max_x) & (y2 == max_y)] = -1

    # Get full coordinates of intersection points
    # for x = 0 (left side of image)
    intersections_1 = np.stack([np.zeros_like(y1), y1], axis=-1)
    # for x = max_x (right side of image)
    intersections_2 = np.stack([np.full_like(y2, max_x), y2], axis=-1)
    # for y = 0 (top side of image)
    intersections_3 = np.stack([x1, np.zeros_like(x1)], axis=-1)
    # for y = max_y (bottom side of image)
    intersections_4 = np.stack([x2, np.full_like(x2, max_y)], axis=-1)

    # Merge all intersections
    intersections = np.stack(
        [intersections_1, intersections_2, intersections_3, intersections_4], axis=-2
    )

    # Check if intersection points are inside image
    # for x = 0 (left side of image)
    is_inside_image_1 = (0 <= intersections_1[:, :, 1]) & (
        intersections_1[:, :, 1] <= max_y
    )
    # for x = max_x (right side of image)
    is_inside_image_2 = (0 <= intersections_2[:, :, 1]) & (
        intersections_2[:, :, 1] <= max_y
    )
    # for y = 0 (top side of image)
    is_inside_image_3 = (0 <= intersections_3[:, :, 0]) & (
        intersections_3[:, :, 0] <= max_x
    )
    # for y = max_y (bottom side of image)
    is_inside_image_4 = (0 <= intersections_4[:, :, 0]) & (
        intersections_4[:, :, 0] <= max_x
    )

    # Merge all checks
    is_inside_image = np.stack(
        [is_inside_image_1, is_inside_image_2, is_inside_image_3, is_inside_image_4],
        axis=-1,
    )

    return intersections, is_inside_image


def getLinePixels(start, end):
    d0 = end[0] - start[0]
    d1 = end[1] - start[1]
    if np.abs(d0) > np.abs(d1):
        x = np.arange(start[0], end[0] + np.sign(d0), np.sign(d0), dtype=np.int32)
        n = len(x)
        y = np.linspace(start[1], end[1], n, endpoint=True, dtype=np.int32)
        return (y, x)
    else:
        y = np.arange(start[1], end[1] + np.sign(d1), np.sign(d1), dtype=np.int32)
        n = len(y)
        x = np.linspace(start[0], end[0], n, endpoint=True, dtype=np.int32)
        return (y, x)


def calculateRayPixels(current_intersections, current_is_inside_image):
    # Find which intersections are correct
    end_points = current_intersections[current_is_inside_image]

    if end_points.shape == 0 or end_points.shape[0] == 0:
        # If no intersections are correct then ray is entirely outside image
        return tuple(([], []))

    if end_points.shape[0] == 1:
        # If only one intersection is correct then ray pass only through 1 pixel
        x, y = end_points[0]
        return tuple(([int(y)], [int(x)]))

    # Calculate pixels of image that are passed through between given end_points
    return getLinePixels(end_points[0], end_points[1])


def processRays(rays_info):
    (emitter_positions, detector_positions, all_scans_rays) = rays_info
    (emitters_x, emitters_y) = zip(*(emitter_positions))
    (detectors_x, detectors_y) = zip(*(detector_positions))

    rays_x = []
    rays_y = []
    for ray in all_scans_rays:
        y, x = ray
        rays_x.extend(np.array(x))
        rays_y.extend(np.array(y))

    rays_info = (emitters_x, emitters_y, detectors_x, detectors_y, rays_x, rays_y)

    return rays_info


def processImage(
    image_data,
    hasToBeTruncated=True,
    hasToBeFiltered=True,
):
    processed_image_data = image_data.copy()
    if hasToBeFiltered:
        kernel_filter_size = 5
        main = 1 / 25
        temp = 1 / 25
        kernel_filter = np.full((kernel_filter_size, kernel_filter_size), temp)
        kernel_filter[kernel_filter_size // 2, kernel_filter_size // 2] = main
        processed_image_data = convolve(
            processed_image_data, kernel_filter, mode="nearest"
        )
    if hasToBeTruncated:
        n_th_percentile = np.percentile(processed_image_data, 99)
        n_th_percentile2 = np.percentile(processed_image_data, 1)
        processed_image_data[processed_image_data > n_th_percentile] = n_th_percentile
        processed_image_data[processed_image_data < n_th_percentile2] = n_th_percentile2

    return processed_image_data


def displayState(
    plot_info,
    processed_rays_info,
    sinogram,
    reconstructed_image,
    mse_value,
    scan_number,
):
    (image_pixels, plot_xlim, plot_ylim) = plot_info
    (emitters_x, emitters_y, detectors_x, detectors_y, rays_x, rays_y) = (
        processed_rays_info
    )

    # Display rays
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].axis("equal")
    axes[0].set_xlim(plot_xlim)
    axes[0].set_ylim(plot_ylim)
    axes[0].plot(emitters_x, emitters_y, "ro", markersize=2)
    axes[0].plot(detectors_x, detectors_y, "go", markersize=2)
    axes[0].plot(rays_x, rays_y, "bo", markersize=1, alpha=0.1)
    axes[0].imshow(image_pixels)
    axes[0].set_title(f"Rays during scan nr. {scan_number}")

    # Display sinogram
    axes[1].imshow(sinogram, cmap="gray")
    axes[1].set_title(f"Sinogram after scan nr. {scan_number}")

    # Display reconstructed image
    axes[2].imshow(reconstructed_image, cmap="gray")
    axes[2].set_title(
        f"Image reconstructed after scan nr. {scan_number}\n mse = {mse_value}"
    )

    clear_output(wait=True)
    plt.show()


def simulateComputedTomographyScan(
    image_filename: str,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
    has_to_use_sinogram_filter: bool = True,
    has_to_save_intermediate_steps: bool = False,
    has_to_show_intermediate_steps: bool = False,
):
    # Load input image
    image = load_bitmap(image_filename)
    image.load()
    image_size = image.size
    image_width, image_height = image_size
    image_data = np.asarray(image, dtype=np.float64)
    image_pixels = image_data.mean(axis=-1)
    image.close()

    # Inscribe image in circle ( center of circle is center of image )
    # Circle pass through centers of 4 corner pixels of image
    # So its radius is equal to half of diagonal of image reduced by 1 pixel in width and height
    diagonal_length = sqrt(pow(image_width - 1, 2) + pow(image_height - 1, 2))
    radius = diagonal_length / 2

    # Calculate position of center of pixel (0,0) of image in a emitters and detectors coordinate system
    image_start_x = -image_width // 2 + 0.5
    image_start_y = -image_height // 2 + 0.5
    image_start = np.array([image_start_x, image_start_y])

    # Calcualate size of plot with rays
    plot_xlim = (
        -floor(radius * 1.1) - image_start_x,
        ceil(radius * 1.1) - image_start_x,
    )
    plot_ylim = (
        ceil(radius * 1.1) - image_start_y,
        -floor(radius * 1.1) - image_start_x,
    )
    plot_info = (image_data / 255, plot_xlim, plot_ylim)

    # Calculate positions of detectors and emitters on circle
    emitter_positions, detector_positions = calculateEmittersDetectorsPositions(
        image_start,
        radius,
        scans_count,
        detectors_count,
        alfa_step,
        detectors_angular_aperture,
    )

    # Calculate intersections and checks which intersections are inside image
    intersections, is_inside_image = calcualateIntesections(
        image_width, image_height, emitter_positions, detector_positions
    )

    # Create matrix of ray pixels
    all_scans_rays = np.empty((scans_count, detectors_count), dtype=np.object_)

    # Create matrix for sinogram data
    sinogram = np.zeros((scans_count, detectors_count), dtype=np.float64)

    # Create kernel used for filtering if needed
    if has_to_use_sinogram_filter:
        kernel_size = 161
        while detectors_count < kernel_size:
            kernel_size = ((kernel_size - 1) // 2) + 1
        kernel_mid_point_index = (kernel_size - 1) // 2
        kernel = np.zeros(kernel_size)
        kernel[kernel_mid_point_index] = 1
        numerator = -4 / (pi**2)
        for k in range(1, (kernel_mid_point_index) + 1, 2):
            kernel_value = (numerator) / (k**2)
            kernel[k + kernel_mid_point_index] = kernel_value
            kernel[-k + kernel_mid_point_index] = kernel_value

    # Create matrix for image data
    image_data = np.zeros((image_height, image_width))

    # Create array of created images
    if has_to_save_intermediate_steps:
        images = np.empty((scans_count, image_height, image_width))

    for scan_number in range(scans_count):
        for detector_number in range(detectors_count):
            # Edge case when emitter and detector is in same position (pi rad detectors_angular_aperture case)
            if (
                emitter_positions[scan_number, detector_number][0]
                - detector_positions[scan_number, detector_number][0]
                < 0.0001
            ) and (
                emitter_positions[scan_number, detector_number][1]
                - detector_positions[scan_number, detector_number][1]
                < 0.0001
            ):
                all_scans_rays[scan_number, detector_number] = tuple(([], []))
                continue
            # Calculate pixels of image that are passed through between given end_points
            pixels_coords = calculateRayPixels(
                intersections[scan_number, detector_number],
                is_inside_image[scan_number, detector_number],
            )

            all_scans_rays[scan_number, detector_number] = pixels_coords

            current_scan_ray = all_scans_rays[scan_number, detector_number]
            # Check if ray passed through any image pixels
            if len(current_scan_ray[0]) == 0:
                sinogram[scan_number, detector_number] = 0
                continue

            # Calculate average of pixels passed by ray
            sinogram[scan_number, detector_number] = np.mean(
                image_pixels[current_scan_ray]
            )

        # Calculate convolution of sinogram and kernel if needed
        if has_to_use_sinogram_filter:
            sinogram[scan_number] = np.convolve(
                sinogram[scan_number], kernel, mode="same"
            )

        for detector_number in range(detectors_count):
            current_scan_ray = all_scans_rays[scan_number, detector_number]
            # Check if ray passed through any image pixels
            if len(current_scan_ray[0]) == 0:
                continue

            # Add value from sinogram to all pixels passed by ray
            image_data[current_scan_ray] += sinogram[scan_number, detector_number]

        # Save image data
        if has_to_save_intermediate_steps:
            images[scan_number] = image_data.copy()

        # Display optional intermediate stages
        if has_to_show_intermediate_steps:
            rays_info = (
                emitter_positions[scan_number],
                detector_positions[scan_number],
                all_scans_rays[scan_number],
            )
            processed_rays_info = processRays(rays_info)
            processed_image = processImage(image_data)
            mse_value = mse(plot_info[0].mean(-1), processed_image)
            displayState(
                plot_info,
                processed_rays_info,
                sinogram,
                processed_image,
                mse_value,
                scan_number + 1,
            )

    if not has_to_show_intermediate_steps:
        rays_info = (
            emitter_positions[scan_number],
            detector_positions[scan_number],
            all_scans_rays[scan_number],
        )
        processed_rays_info = processRays(rays_info)
        processed_image = processImage(image_data)
        mse_value = mse(plot_info[0].mean(-1), processed_image)
        displayState(
            plot_info,
            processed_rays_info,
            sinogram,
            processed_image,
            mse_value,
            scan_number + 1,
        )

    if has_to_save_intermediate_steps:
        all_rays_info = (emitter_positions, detector_positions, all_scans_rays)
        return (plot_info, all_rays_info, sinogram, images, scans_count)
    else:
        rays_info = (
            emitter_positions[scan_number],
            detector_positions[scan_number],
            all_scans_rays[scan_number],
        )
        return (plot_info, rays_info, sinogram, image_data, 1)


def showStateAfterScan(states, scan, hasToBeTruncated=True, hasToBeFiltered=True):
    plot_info, all_rays_info, sinogram, image_data, states_number = states
    if states_number == 1:
        processed_rays_info = processRays(all_rays_info)
        processed_image = processImage(image_data, hasToBeTruncated, hasToBeFiltered)

        mse_value = mse(plot_info[0].mean(-1), processed_image)
        displayState(
            plot_info, processed_rays_info, sinogram, processed_image, mse_value, scan
        )
    else:
        emitter_positions, detector_positions, all_scans_rays = all_rays_info

        rays_info = (
            emitter_positions[scan - 1],
            detector_positions[scan - 1],
            all_scans_rays[scan - 1],
        )
        processed_rays_info = processRays(rays_info)

        new_sinogram = np.zeros_like(sinogram)
        new_sinogram[:scan] = sinogram[:scan]

        processed_image = processImage(
            image_data[scan - 1], hasToBeTruncated, hasToBeFiltered
        )
        mse_value = mse(plot_info[0].mean(-1), processed_image)
        displayState(
            plot_info,
            processed_rays_info,
            new_sinogram,
            processed_image,
            mse_value,
            scan,
        )


@dataclass
class PatientInfo:
    first_name: str
    surname: str
    identifier: str
    birth_date: datetime.date


def create_dicom_from_image(
    image,
    output_path,
    patient_info: PatientInfo,
    study_datetime: datetime,
    series_description="",
    private_comment: Optional[None] = None,
):
    if isinstance(image, Image.Image):
        image = np.array(image)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.PatientName = f"{patient_info.surname}^{patient_info.first_name}"
    ds.PatientID = patient_info.identifier
    ds.PatientBirthDate = patient_info.birth_date.strftime("%Y%m%d")
    ds.Modality = "CT"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate = study_datetime.strftime("%Y%m%d")
    ds.StudyTime = study_datetime.strftime("%H%M%S")
    ds.Rows, ds.Columns = image.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.SeriesDescription = series_description
    ds.add_new_private("Comments", 0x000B, 0x01, private_comment, "UT")

    ds.PixelData = image.astype(np.uint16).tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)


def mse(original: np.ndarray, recreated: np.ndarray):
    original_min = original.min()
    original_max = original.max()
    original_normalized = (original - original_min) / (original_max - original_min)
    recreated_min = recreated.min()
    recreated_max = recreated.max()
    recreated_normalized = (recreated - recreated_min) / (recreated_max - recreated_min)
    return np.mean((recreated_normalized - original_normalized) ** 2)
