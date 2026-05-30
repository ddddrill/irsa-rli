import math
import logging
import os

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from models.radar import Radar
from models.target import Target
from processing.isar_processor import StandardISARProcessor, PolarISARProcessor

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 3e8
NUM_IMAGES = 16


class DataProcessor(QThread):
    """Вычислительный поток: формирование тензора РЛИ из данных цели.

    Оркестратор: связывает Радар, Цель и Процессор.
    """

    frame_progress = pyqtSignal(int)
    frame_saved = pyqtSignal(str, int)

    def __init__(self, filename, method, f_c, Xmax, Ymax, spectr_w, ang, range_m,
                 nifft_size=1024, save_dir=None):
        super().__init__()
        self.filename = filename
        self.method = method
        self.f_c = f_c * 1e9
        self.Xmax = Xmax
        self.Ymax = Ymax
        self.spectr_w = spectr_w * 1e9
        self.ang = ang
        self.range_m = range_m
        self.nifft_size = nifft_size
        self.save_dir = save_dir
        self._is_running = True

        self.radar = Radar(
            c=SPEED_OF_LIGHT,
            f_c=self.f_c,
            Xmax=self.Xmax,
            Ymax=self.Ymax,
            spectr_w=self.spectr_w,
            ph_c=0.0,
            ang=self.ang,
            nifft_size=self.nifft_size,
        )
        self.v_cos, self.v_sin, self.v_exp, self.v_complx, self.v_floor = (
            self.radar.vectorize_functions()
        )

    def run(self):
        self._compute_and_save()

    def stop(self):
        self._is_running = False

    def compute_single(self):
        """Вычислить один первый кадр синхронно (без потока)."""
        target = Target(self.filename)
        intens, x, y = target.get_frame(0)
        x_1 = x - math.floor((max(x) - min(x)) / 2)
        y_1 = y - math.floor((max(y) - min(y)) / 2)
        return self.radioimage_single(intens, x_1, y_1)

    def _compute_and_save(self):
        if self.save_dir and os.path.exists(self.save_dir):
            for item in os.listdir(self.save_dir):
                item_path = os.path.join(self.save_dir, item)
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)

        target = Target(self.filename, pri=1.0, ang_rad=math.radians(self.ang))

        for i in range(NUM_IMAGES):
            if not self._is_running:
                break
            intens, x, y = target.get_frame(i)
            x_1 = x - math.floor((max(x) - min(x)) / 2)
            y_1 = y - math.floor((max(y) - min(y)) / 2)

            frame_data = self.radioimage_single(intens, x_1, y_1)

            if self.save_dir:
                self._save_frame(i + 1, frame_data)

            self.frame_progress.emit(i + 1)
            logger.info("Обработан кадр %d/%d", i + 1, NUM_IMAGES)

    def _save_frame(self, frame_num, mat):
        frame_dir = os.path.join(self.save_dir, f"frame_{frame_num:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        from PIL import Image

        if self.method == "стандартный":
            field_data = abs(np.rot90(mat[0], 2))
            img_data = abs(np.rot90(mat[3], 2))

            field_norm = (field_data / field_data.max() * 255).astype(np.uint8)
            img_norm = (img_data / img_data.max() * 255).astype(np.uint8)

            Image.fromarray(field_norm).save(os.path.join(frame_dir, "field.png"))
            Image.fromarray(img_norm).save(os.path.join(frame_dir, "rli.png"))

        elif self.method == "с полярным переформатированием":
            field_data = abs(mat[0])
            img_data = abs(mat[3] / (mat[4] * mat[5]))

            field_norm = (field_data / field_data.max() * 255).astype(np.uint8)
            img_norm = (img_data / img_data.max() * 255).astype(np.uint8)

            Image.fromarray(field_norm).save(os.path.join(frame_dir, "field.png"))
            Image.fromarray(img_norm).save(os.path.join(frame_dir, "rli.png"))

        self.frame_saved.emit(frame_dir, frame_num)

    def radioimage_single(self, intens, x_coord, y_coord):
        if self.method == "стандартный":
            return self._standard_process(intens, x_coord, y_coord)
        elif self.method == "с полярным переформатированием":
            return self._polar_process(intens, x_coord, y_coord)

    def _standard_process(self, intens, x_coord, y_coord):
        Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )

        Es = proc.compute_field(intens=intens, x=x_coord, y=y_coord)
        base_img, base_x, base_y = proc.compute_image(Es)
        return Es, f_r, ph_r, base_img, base_x, base_y

    def _polar_process(self, intens, x_coord, y_coord):
        Nf, Nph, f_r, ph_r, k_r, Nifft_fr, Nifft_ph, df, dph, FR, PH = (
            self.radar.base_img_params()
        )

        base_proc = StandardISARProcessor(
            Nf=Nf, Nph=Nph,
            f_r=f_r, ph_r=ph_r, k_r=k_r,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph,
            df=df, dph=dph, FR=FR, PH=PH,
            f_c=self.f_c,
            complex_v=self.v_complx, exp_v=self.v_exp,
        )
        Es = base_proc.compute_field(intens=intens, x=x_coord, y=y_coord)

        Nf, Nph, kx, ky, kxMax, kyMax, kxMin, kyMin, Nifft_fr, Nifft_ph, M = (
            self.radar.polar_img_params()
        )

        polar_proc = PolarISARProcessor(
            Nf=Nf, Nph=Nph,
            kx=kx, ky=ky,
            kxMax=kxMax, kyMax=kyMax, kxMin=kxMin, kyMin=kyMin,
            Nifft_fr=Nifft_fr, Nifft_ph=Nifft_ph, M=M,
        )

        Es_pol, grid_x, grid_y = polar_proc.polar_reformat(Es)
        img_polar, Kx, Ky, len_xp, len_yp = polar_proc.compute_image(
            Es_pol, grid_x, grid_y
        )
        return Es_pol, grid_x, grid_y, img_polar, Kx, Ky, len_xp, len_yp
