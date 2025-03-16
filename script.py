from math import cos, pi, sin, sqrt

import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageFile import ImageFile


def load_bitmap(filename: str) -> ImageFile:
    image = Image.open(filename)
    return image


def scan(
    image,
    scans_count: int,
    alfa_step: float,
    detectors_count: int,
    detectors_angular_aperture: float,
):
    # angle (as string with 2 decimal places) --> list of data
    data: dict[str, list[float]] = {}

    # 1. Opisz okrąg na obrazie
    im_size = image.size
    diagonal_length = sqrt(pow(im_size[0], 2) + pow(im_size[1], 2))
    radius = diagonal_length / 2

    # 2. Wyznacz położenie n emiterów i n detektorów na okręgu
    alfa = alfa_step
    for scan_number in range(scans_count):
        emitters: list[tuple[int, int]] = []
        detectors: list[tuple[int, int]] = []

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
            detector_coords = (round(x_d_i), round(y_d_i))
            detectors.append(detector_coords)

            e_i_arg = alfa + mid_angle_diff
            x_e_i = radius * cos(e_i_arg)
            y_e_i = radius * sin(e_i_arg)
            emitter_coords = (round(x_e_i), round(y_e_i))
            emitters.append(emitter_coords)

        alfa += alfa_step

    x, y = zip(*(detectors + emitters))
    plt.figure(figsize=(8, 8))
    plt.axis("equal")
    plt.plot(x, y, "ro")
    plt.savefig("emitters-detectors.png")

    # 3. Wyznacz linie przejścia
    # 3. a) wyznacz linie przejścia od emiterów do detektorów
    # 3. b) wyznacz linie przejścia tylko przez obraz

    # 4. Wyznacz (średnią) wartość przejścia sygnału

    # 5. Pokaż sinogram
    im_pixels = image.load()


def main():
    image = load_bitmap("./obrazy/Kropka.jpg")
    print(image)
    scan(image, 1, 0.3 * pi, 13, 0.2 * pi)
    # image.show()


# Aplikacja powinna móc pozwolić konfigurować następujące elementy:
# Krok ∆α układu emiter/detektor.
# Dla jednego układu emiter/detektor liczbę detektorów (n).
# Rozwartość/rozpiętość układu emiter/detektor (l).


if __name__ == "__main__":
    main()
