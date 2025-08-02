import os
import uuid
import time
from PIL import Image, ImageEnhance
from multiprocessing import Pool, Manager
import traceback
from io import BytesIO
from tqdm import tqdm
import sys

# if runnning this ensure folders are correct, these are placeholders
OUTPUT_DIR = 'D:/outputcaptchas'
CUTOUT_SHAPES_DIR = './cutout_images/'
BASE_IMAGES_DIR = './base_images/'

NUM_PROCESSES = 16 
UPDATE_FREQUENCY = 1000 
SWITCH_DIR_INTERVAL = 300 # switching to seperate batches due to i/o writing issues on large folders, quicker to solve it like this
BATCH_DIR_PREFIX = 'batch_'

TARGET_HEIGHT = 200
TARGET_WIDTH = 300
CUTOFF_SIZE = 80 # images are 80x80, ensures that images aren't being placed off of the grid of base image

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp') # future implementation of other types of images

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

CLASS_ID = 0


def create_shape_mask(cutout_shape):
    if cutout_shape.mode != 'RGBA':
        cutout_shape = cutout_shape.convert('RGBA')
    shape_mask = cutout_shape.split()[3]
    thresholded_mask = shape_mask.point(lambda p: 255 if p > 128 else 0, mode='1')
    return thresholded_mask

def fade_cutout_area(base_image, cutout_shape, shape_mask, slider_x, cutout_y):
    try:
        if base_image.mode != "RGBA":
            base_image = base_image.convert("RGBA")

        area_box = (slider_x, cutout_y, slider_x + cutout_shape.width, cutout_y + cutout_shape.height)
        
        base_width, base_height = base_image.size
        area_box = (
            max(0, area_box[0]),
            max(0, area_box[1]),
            min(base_width, area_box[2]),
            min(base_height, area_box[3])
        )

        cutout_region = (
            max(0, -slider_x),
            max(0, -cutout_y),
            cutout_shape.width - max(0, slider_x + cutout_shape.width - base_width),
            cutout_shape.height - max(0, cutout_y + cutout_shape.height - base_height)
        )

        area = base_image.crop(area_box)
        mask_cropped = shape_mask.crop(cutout_region).resize(area.size, Image.NEAREST)

        enhancer = ImageEnhance.Brightness(area)
        faded_area = enhancer.enhance(0.4) 

        base_image.paste(faded_area, area_box, mask_cropped)
        
    except Exception as e:
        traceback.print_exc()
        print(f"Cannot fade ({slider_x}, {cutout_y}): {e}")
        raise

    return base_image

def reconstruct_image(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert('RGBA')

def generate_unique_filename(base_image_name, x, y, cutout_index):
    # legacy code for debugging so I kept it in, it's just naming images can be simplified if necessary
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{base_image_name}_x{x}_y{y}_cutout{cutout_index}_{unique_id}.png"
    return filename

def worker(args):
    (tasks_queue, base_images_bytes, cutout_shapes, lock, thread_index, batch_dir_lock) = args
    captchas_generated = 0
    skipped_tasks = 0 # another legacy debugger, left it in here for any weird exceptions
    log_entries = []
    start_time = time.time()
    batch_index = 1
    last_switch_time = start_time

    current_batch_dir = os.path.join(OUTPUT_DIR, f'{BATCH_DIR_PREFIX}{batch_index}')
    os.makedirs(current_batch_dir, exist_ok=True)
    log_filename = os.path.join(current_batch_dir, f'slider_positions_{thread_index}.txt')

    while True:
        try:
            task = tasks_queue.get_nowait()
        except:
            break
        base_image_filename, x, y, cutout_index = task

        current_time = time.time()
        if current_time - last_switch_time >= SWITCH_DIR_INTERVAL:
            with batch_dir_lock:
                batch_index += 1
            current_batch_dir = os.path.join(OUTPUT_DIR, f'{BATCH_DIR_PREFIX}{batch_index}')
            os.makedirs(current_batch_dir, exist_ok=True)
            log_filename = os.path.join(current_batch_dir, f'slider_positions_{thread_index}.txt')
            last_switch_time = current_time

        try:
            base_image_bytes = base_images_bytes.get(base_image_filename)
            if not base_image_bytes:
                skipped_tasks += 1
                continue 
            base_image_original = reconstruct_image(base_image_bytes)

            cutout_shape_data = cutout_shapes[cutout_index]
            cutout_filename, cutout_shape, shape_mask = cutout_shape_data

            base_image = base_image_original.copy()

            base_image_with_faded_cutout = fade_cutout_area(base_image, cutout_shape, shape_mask, x, y)

            base_image_name = os.path.splitext(base_image_filename)[0]
            captcha_filename = generate_unique_filename(base_image_name, x, y, cutout_index)
            captcha_path = os.path.join(current_batch_dir, captcha_filename)

            base_image_with_faded_cutout.convert('RGB').save(captcha_path, format='PNG', optimize=True)

            slider_x_center = x + (CUTOFF_SIZE / 2)
            slider_y_center = y + (CUTOFF_SIZE / 2)
            slider_position = (slider_x_center, slider_y_center)

            log_entries.append(f"{captcha_filename},{slider_x_center},{slider_y_center}\n")
            captchas_generated += 1

            label_filename = os.path.splitext(captcha_filename)[0] + '.txt'
            label_path = os.path.join(current_batch_dir, label_filename)

            x_center_norm = slider_x_center / TARGET_WIDTH
            y_center_norm = slider_y_center / TARGET_HEIGHT
            width_norm = CUTOFF_SIZE / TARGET_WIDTH
            height_norm = CUTOFF_SIZE / TARGET_HEIGHT

            with open(label_path, 'w') as label_file:
                label_file.write(f"{CLASS_ID} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

            if captchas_generated % UPDATE_FREQUENCY == 0:
                with lock:
                    with open(log_filename, 'a') as log_file:
                        log_file.writelines(log_entries)
                        log_entries = []

        except Exception as e:
            traceback.print_exc()
            print(f"Error task {task}: {e}")
            skipped_tasks += 1
            continue

    if log_entries:
        with lock:
            with open(log_filename, 'a') as log_file:
                log_file.writelines(log_entries)

    return captchas_generated, skipped_tasks


def generate_captchas():
    manager = Manager()
    lock = manager.Lock()
    batch_dir_lock = manager.Lock()

    base_images_filenames = [
        f for f in os.listdir(BASE_IMAGES_DIR)
        if os.path.isfile(os.path.join(BASE_IMAGES_DIR, f)) and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print(f"Total base images found: {len(base_images_filenames)}")
    if not base_images_filenames:
        print("No base images found. Exiting.")
        sys.exit(1)

    base_images_bytes = manager.dict()
    for filename in base_images_filenames:
        base_image_path = os.path.join(BASE_IMAGES_DIR, filename)
        try:
            with Image.open(base_image_path) as img:
                img = img.convert('RGBA')
                if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.ANTIALIAS)
                with BytesIO() as buffer:
                    img.save(buffer, format='PNG')
                    base_images_bytes[filename] = buffer.getvalue()
        except Exception as e:
            print(f"Error loading base image {filename}: {e}")
            continue

    print(f"Total base images loaded: {len(base_images_bytes)}")

    cutout_shape_filenames = [
        f for f in os.listdir(CUTOUT_SHAPES_DIR)
        if os.path.isfile(os.path.join(CUTOUT_SHAPES_DIR, f)) and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print(f"Total cutout shapes found: {len(cutout_shape_filenames)}")
    if not cutout_shape_filenames:
        print("No cutout shapes found. Ensure paths are set properly.")
        sys.exit(1)

    cutout_shapes = []
    for idx, filename in enumerate(cutout_shape_filenames):
        cutout_path = os.path.join(CUTOUT_SHAPES_DIR, filename)
        try:
            with Image.open(cutout_path) as img:
                img = img.convert('RGBA')
                shape_mask = create_shape_mask(img)
                if img.size != (CUTOFF_SIZE, CUTOFF_SIZE):
                    img = img.resize((CUTOFF_SIZE, CUTOFF_SIZE), Image.ANTIALIAS)
                    shape_mask = shape_mask.resize((CUTOFF_SIZE, CUTOFF_SIZE), Image.NEAREST)
                cutout_shapes.append((filename, img.copy(), shape_mask.copy()))
        except Exception as e:
            print(f"Error loading cutout shape {filename}: {e}")
            continue

    print(f"Total cutout shapes loaded: {len(cutout_shapes)}")
    if len(cutout_shapes) < 9:
        print("9 main shapes did not load, debug needed")

    tasks = []
    y_min = 30
    y_max = 170 - CUTOFF_SIZE + 1 
    for base_image_filename in base_images_bytes.keys():
        for y in range(y_min, y_max):
            for x in range(0, TARGET_WIDTH - CUTOFF_SIZE + 1):
                for cutout_index in range(len(cutout_shapes)):
                    tasks.append((base_image_filename, x, y, cutout_index))
    total_tasks = len(tasks)
    print(f"Total CAPTCHAs to generate: {total_tasks}")

    tasks_queue = manager.Queue()
    for task in tasks:
        tasks_queue.put(task)

    worker_args = []
    for i in range(NUM_PROCESSES):
        worker_args.append((
            tasks_queue,
            base_images_bytes,
            cutout_shapes,
            lock,
            i,
            batch_dir_lock
        ))

    print(f"Starting {NUM_PROCESSES} threads.")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(worker, worker_args), total=len(worker_args), desc="Generating CAPTCHAs"))

    total_captchas = sum([result[0] for result in results])
    total_skipped = sum([result[1] for result in results])

    print(f"CAPTCHA generation completed.")
    print(f"Total CAPTCHAs generated: {total_captchas}")
    print(f"Total tasks skipped due to errors: {total_skipped}")

if __name__ == "__main__":
    generate_captchas()
