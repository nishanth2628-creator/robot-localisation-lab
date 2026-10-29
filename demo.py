"""Reproduce the ten-seed localisation study and save metrics + PNG/SVG figures."""

import argparse
from html import escape
import json
import math
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from localization import estimate, generate_dataset, run_study


class Figure:
    """Draw the same basic vector elements into PNG and SVG outputs."""

    def __init__(self, width=1440, height=1340):
        self.image = Image.new('RGB', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<title>Robot Localisation Lab: wheel odometry and landmark EKF</title>',
                    '<desc>A seeded trajectory comparison and position-error chart, with results across ten paired noise seeds.</desc>',
                    f'<rect width="{width}" height="{height}" fill="white"/>']

    def font(self, size, bold):
        key = (size, bold)
        if key not in self.fonts:
            for name in ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
                         'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'):
                try:
                    self.fonts[key] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    pass
            else:
                self.fonts[key] = ImageFont.load_default(size=size)
        return self.fonts[key]

    def text(self, x, y, value, size=19, fill='#17273B', bold=False, anchor='start'):
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{fill}">{escape(str(value))}</text>')
        self.draw.text((x, y), str(value), font=self.font(size, bold), fill=fill,
                       anchor={'start': 'ls', 'middle': 'ms', 'end': 'rs'}[anchor])

    def rect(self, bounds, fill, stroke=None, width=1):
        x1, y1, x2, y2 = bounds
        self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
        self.draw.rectangle(bounds, fill=fill, outline=stroke, width=width)

    def line(self, points, fill, width=1):
        points = [(float(x), float(y)) for x, y in points]
        self.svg.append(f'<polyline points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in points)}" fill="none" stroke="{fill}" stroke-width="{width}" stroke-linejoin="round"/>')
        self.draw.line(points, fill=fill, width=width, joint='curve')

    def circle(self, x, y, radius, fill, stroke=None, width=1):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=stroke, width=width)

    def save(self, destination):
        self.image.save(destination / 'demo.png')
        (destination / 'demo.svg').write_text('\n'.join(self.svg + ['</svg>']), encoding='utf-8')


def draw_demo(data, runs, results, destination):
    f = Figure()
    navy, muted = '#17273B', '#5C6E82'
    blue, teal, orange = '#3877B7', '#00877B', '#C26916'
    colors = {'odometry_only': blue, 'landmark_ekf': teal, 'dropout_ekf': orange}
    labels = {'odometry_only': 'Wheel odometry only', 'landmark_ekf': 'Landmark EKF',
              'dropout_ekf': 'EKF with 20-40 s dropout'}
    f.text(80, 58, 'Robot Localisation Lab', size=36, bold=True)
    f.text(80, 96, 'Noisy wheel odometry + known landmark range/bearing | 2D pose EKF', size=22, fill=muted)
    f.text(80, 137, 'ONE SYNTHETIC ROUTE | 10 PAIRED NOISE SEEDS | 60 SECONDS', size=15, bold=True, fill=muted)
    x0, y0, size = 80, 175, 730
    bounds = (-4., 20., -12., 12.)

    def point(x, y):
        return x0+(x-bounds[0])/(bounds[1]-bounds[0])*size, y0+size-(y-bounds[2])/(bounds[3]-bounds[2])*size

    f.rect((x0, y0, x0+size, y0+size), '#F7F9FC')
    for value in range(-4, 21, 4):
        x, _ = point(value, 0)
        f.line([(x, y0), (x, y0+size)], '#DFE7EF')
        f.text(x, y0+size+26, str(value), size=15, fill=muted, anchor='middle')
    for value in range(-12, 13, 4):
        _, y = point(0, value)
        f.line([(x0, y), (x0+size, y)], '#DFE7EF')
        f.text(x0-17, y+5, str(value), size=15, fill=muted, anchor='end')
    f.text(x0+size+22, y0+size+26, 'x (m)', size=15, fill=muted)
    f.text(x0-22, y0-17, 'y (m)', size=15, fill=muted)
    for index, landmark in enumerate(data.landmarks):
        x, y = point(*landmark)
        f.rect((x-6, y-6, x+6, y+6), '#CFD8E2', '#8294A7')
        f.text(x+11, y-8, f'L{index}', size=13, fill=muted)
    f.line([point(*pose[:2]) for pose in data.truth], navy, width=5)
    for name in ('odometry_only', 'dropout_ekf', 'landmark_ekf'):
        f.line([point(*pose[:2]) for pose in runs[name].states], colors[name], width=3 if name != 'dropout_ekf' else 2)
    sx, sy = point(*data.truth[0, :2])
    f.circle(sx, sy, 9, 'white', navy, 2)
    f.text(sx-17, sy+24, 'Start', size=15, fill=navy)
    ex, ey = point(*data.truth[-1, :2])
    f.circle(ex, ey, 9, 'white', navy, 2)
    f.text(ex+18, ey+5, 'True finish', size=15)
    f.text(x0, y0+size+55, 'Shown trajectories use seed 0; truth and EKF largely overlap.', size=17, fill=muted)

    panel_x = 900
    f.text(panel_x, 193, 'MEAN POSITION RMSE ACROSS 10 SEEDS', size=16, bold=True, fill=muted)
    for index, name in enumerate(('odometry_only', 'landmark_ekf', 'dropout_ekf')):
        row = results['aggregate'][name]
        y = 244 + index * 167
        f.text(panel_x, y, labels[name], size=23, bold=True, fill=colors[name])
        f.text(panel_x, y+47, f'{row["mean_position_rmse_m"]:.3f} m', size=37, bold=True)
        f.text(panel_x, y+79, f'Sample std: {row["sample_std_position_rmse_m"]:.3f} m', size=18, fill=muted)
        f.text(panel_x, y+106, f'Mean final error: {row["mean_final_position_error_m"]:.3f} m', size=18, fill=muted)
    f.line([(panel_x, 725), (1360, 725)], '#DFE7EF')
    f.text(panel_x, 757, 'Same initial estimate and measured encoders.', size=18)
    f.text(panel_x, 786, 'Wheel scale bias: +1.2% left / -0.8% right.', size=18)
    f.text(panel_x, 815, 'Known landmark positions and associations.', size=18)
    f.line([(panel_x, 851), (panel_x+40, 851)], navy, width=5)
    f.text(panel_x+55, 858, 'Ground truth (scoring only)', size=18)
    f.rect((panel_x, 879, panel_x+17, 896), '#CFD8E2', '#8294A7')
    f.text(panel_x+32, 895, 'Known landmark', size=18)
    f.text(panel_x, 935, 'Synthetic experiment; no hardware validation.', size=17, fill=muted)

    f.text(80, 1000, 'POSITION ERROR OVER TIME | SEED 0 | LOG SCALE', size=16, bold=True, fill=muted)
    px, py, pw, ph = 80, 1030, 1280, 210
    f.rect((px, py, px+pw, py+ph), '#F7F9FC')
    f.rect((px+20/60*pw, py, px+40/60*pw, py+ph), '#FFF0DE')
    # Fixed axes let regenerated figures remain comparable across runs.
    low, high = 0.001, 100.0

    def error_y(value):
        value = min(high, max(low, float(value)))
        return py+ph-(math.log10(value)-math.log10(low))/(math.log10(high)-math.log10(low))*ph

    for value in (0.001, 0.01, 0.1, 1, 10, 100):
        y = error_y(value)
        f.line([(px, y), (px+pw, y)], '#DFE7EF')
        f.text(px-14, y+5, f'{value:g}', size=14, fill=muted, anchor='end')
    times = np.arange(len(data.truth)) * data.dt
    for time in range(0, 61, 10):
        x = px+time/60*pw
        f.line([(x, py), (x, py+ph)], '#DFE7EF')
        f.text(x, py+ph+27, str(time), size=15, fill=muted, anchor='middle')
    for name in ('odometry_only', 'dropout_ekf', 'landmark_ekf'):
        errors = np.linalg.norm(runs[name].states[:, :2] - data.truth[:, :2], axis=1)
        f.line([(px+time/60*pw, error_y(error)) for time, error in zip(times, errors)], colors[name], 2)
    f.text(px-9, py-13, 'm', size=14, fill=muted, anchor='end')
    f.text(px+pw, py+ph+52, 'time (s)', size=15, fill=muted, anchor='end')
    f.text(px+30/60*pw, py+24, 'All landmark observations masked', size=16, fill=orange, anchor='middle')
    f.text(80, 1320, 'RMSE measures this route only; sample spread is not a confidence interval or a consistency guarantee.', size=17, fill=muted)
    f.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_study()
    results['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__}
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    data = generate_dataset(0)
    runs = {'odometry_only': estimate(data, use_landmarks=False), 'landmark_ekf': estimate(data),
            'dropout_ekf': estimate(data, dropout=(20., 40.))}
    draw_demo(data, runs, results, args.output)
    print(json.dumps(results['aggregate'], indent=2))
    print(f'Saved metrics.json, demo.png and demo.svg to {args.output.resolve()}')


if __name__ == '__main__':
    main()
