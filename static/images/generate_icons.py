import os
import cairosvg

input_svg = "mg.svg"

sizes = [48, 72, 96, 128, 192, 256, 384, 512]

output_dir = "icons"
os.makedirs(output_dir, exist_ok=True)

for size in sizes:
    output_path = os.path.join(output_dir, f"mg-{size}.png")
    cairosvg.svg2png(url=input_svg, write_to=output_path, output_width=size, output_height=size)
    print(f"[✔] Generated: {output_path}")
