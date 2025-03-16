from math import cos, pi, sin, sqrt
from statistics import mean

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.draw import line_nd


def load_bitmap(filename: str) -> ImageFile:
    image = Image.open(filename)
    return image


def scan(
    image,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
) -> dict[float, list[float]]:
    # angle --> list of data
    data: dict[float, list[float]] = {}

    # 1. Opisz okrąg na obrazie
    im_size = image.size
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2

    im_left_x = -im_size[0] / 2
    im_right_x = im_size[0] / 2
    im_top_y = -im_size[1] / 2
    im_bottom_y = im_size[1] / 2

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu

    alfa = alfa_step
    for scan_number in range(scans_count):
        print(f"Scan {scan_number}")
        data[alfa] = []
        emitters: list[tuple[int, int]] = []
        detectors: list[tuple[int, int]] = []

        pixels_on_scan_lines_within_image: list[tuple[int, int]] = []
        for detector_number in range(detectors_count):
            d_i_arg = (
                alfa
                + pi
                - detectors_angular_aperture / 2
                + detector_number * detectors_angular_aperture / (detectors_count - 1)
            )
            mid_angle = alfa + pi
            mid_angle_diff = mid_angle - d_i_arg
            x_d_i = radius * cos(d_i_arg)
            y_d_i = radius * sin(d_i_arg)
            detector_coords = (x_d_i, y_d_i)
            detectors.append(detector_coords)

            e_i_arg = alfa + mid_angle_diff
            x_e_i = radius * cos(e_i_arg)
            y_e_i = radius * sin(e_i_arg)
            emitter_coords = (x_e_i, y_e_i)
            emitters.append(emitter_coords)

            # 3. Wyznacz linie przejścia
            # 3. a) wyznacz linie przejścia od emiterów do detektorów
            pixels_coords = line_nd(emitter_coords, detector_coords, endpoint=True)
            for pixel_x, pixel_y in zip(pixels_coords[0], pixels_coords[1]):
                # 3. b) wyznacz linie przejścia tylko przez obraz
                if (
                    im_left_x <= pixel_x <= im_right_x
                    and im_top_y <= pixel_y <= im_bottom_y
                ):
                    pixels_on_scan_lines_within_image.append((pixel_x, pixel_y))

            # 4. Wyznacz (średnią) wartość przejścia sygnału
            im_pixels = image.load()
            pixels_greyscale_values: list[float] = []
            for pixel_x, pixel_y in pixels_on_scan_lines_within_image:
                rgb = im_pixels[pixel_x, pixel_y]
                val = mean(rgb)
                pixels_greyscale_values.append(val)
            whole_line_mean = mean(pixels_greyscale_values)
            data[alfa].append(whole_line_mean)

        alfa += alfa_step

    # 5. Zwróć sinogram (bez filtracji)
    return data

    # Pokaż linie skanowania --> w pętli for (zewnętrznej) dodać break na końcu,
    # by pokazać linie tylko dla jednego skanu
    # x, y = zip(*(pixels_on_scan_lines_within_image + detectors + emitters))
    # plt.figure(figsize=(8, 8))
    # plt.axis("equal")
    # plt.xlim((-floor(radius * 1.1), ceil(radius * 1.1)))
    # plt.ylim((-floor(radius * 1.1), ceil(radius * 1.1)))
    # plt.plot(x, y, "ro")
    # plt.savefig("pixels.png")


def write_matrix_to_grayscale_file(matrix, output_filename: str) -> None:
    height = len(matrix)
    width = len(matrix[0])
    flat_pixels = sum(matrix, [])

    image = Image.new("L", (width, height))  # "L" mode for grayscale
    image.putdata(flat_pixels)

    image.save(output_filename)


def main():
    image = load_bitmap("./obrazy/Kropka.jpg")
    # sinogram = scan(image, 90, 1 / 90 * pi, 180, 0.2 * pi)
    sinogram = scan(image, 50, 2 / 50 * pi, 18, 0.2 * pi)
    sinogram_matrix = list(sinogram.values())
    write_matrix_to_grayscale_file(sinogram_matrix, "kropka_sinogram.png")


# Aplikacja powinna móc pozwolić konfigurować następujące elementy:
# Krok ∆α układu emiter/detektor.
# Dla jednego układu emiter/detektor liczbę detektorów (n).
# Rozwartość/rozpiętość układu emiter/detektor (l).


if __name__ == "__main__":
    main()
