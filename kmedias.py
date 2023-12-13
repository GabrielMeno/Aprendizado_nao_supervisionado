import os
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
class MODE:
    CONTINUE = 0
    RESTART = 1
def kmeans(path, clusters):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    retval, labels, centers = cv2.kmeans(
        pixel_vals,
        clusters,
        None,

        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.85),


        20,

        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    return segmented_image
def mimagem (img):
    height, width, channels = img.shape
    plt.figure()
    plt.imshow(img)
    fig = plt.gcf()
    fig.set_size_inches(width / 100, height / 100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
def contagem(img):
    colors, counts = np.unique(
        img.reshape(-1, img.shape[-1]),
        return_counts=True,
        axis=0,
    )
    return counts.size
def simage(img, name, clusters, output_id=0):
    # Convertendo a imagem de RGB para BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Cria a pasta de saída, caso não exista
    folder = f'./outputs/{output_id}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Salva a imagem
    path = f'{folder}/{name[:-4]}_{clusters}.png'
    if not cv2.imwrite(path, img):
        raise Exception('Erro ao salvar imagem')

    return path
def tamanho(path):
    return os.path.getsize(path) / 1000000
def setup_logger(output_id):
    log_path = f'./outputs/{output_id}/log.txt'

    # Caso não exista, cria o arquivo de log
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(message)s',
        encoding='utf-8',
    )

    # Seta o nível de log para INFO e adiciona um handler para imprimir no console
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
def print_info(path, resolution=False, timer=None, original_size=None):
    logger = logging.getLogger()

    size = tamanho(path)
    width, height, channels = cv2.imread(path).shape

    img = cv2.imread(path)
    colors = contagem(img)

    if resolution:
        logging.info(f'Resolução: {width}x{height}')

    logger.info(f'Numero de cores: {colors}')
    logger.info(f'Tamanho: {size} mb')

    if original_size:
        logger.info(f'Compressao: {100 - (size / original_size * 100):.2f}%')

    if timer:
        logger.info(f'Tempo: {timer:.2f} s')

    logger.info('')
def print_initial_info(image_name):
    logger = logging.getLogger()
    logger.info('=====================================')
    logger.info(f'Imagem: {image_name}')
def get_images():
    return [f for f in os.listdir('images') if f.endswith('.png')]
def get_output_images(output_id):
    return [f for f in os.listdir(f'./outputs/{output_id}') if f.endswith('.png')]
def get_last_output():
    return max([int(f) for f in os.listdir('outputs') if f.isdigit()] + [0])
def output_exists(output_id, image, clusters):
    return os.path.exists(f'./outputs/{output_id}/{image[:-4]}_{clusters}.png')
if __name__ == '__main__':

    # Parâmetros de execução
    mode = MODE.RESTART
    sizes = [1, 3, 5, 9, 13, 19, 23]


    images = get_images()
    last_output = get_last_output()

    # Verifica, caso o modo seja CONTINUE, se todos os outputs já foram gerados
    if mode == MODE.CONTINUE and last_output > 0:
        if len(get_output_images(last_output)) >= len(images) * len(sizes):
            mode = MODE.RESTART

    output_id = last_output + 1 if mode == MODE.RESTART else last_output

    setup_logger(output_id)

    for image_name in images:
        path = f'./images/{image_name}'

        # Faz o log inicial da imagem
        if mode == MODE.RESTART or not output_exists(output_id, image_name, sizes[0]):
            print_initial_info(image_name)
            print_info(path, resolution=True)

        # Salva o tamanho original da imagem para calcular a compressão
        original_size = tamanho(path)

        for n in sizes:

            # Pula a execução caso o output já exista e o modo seja CONTINUE
            if mode == MODE.CONTINUE and output_exists(output_id, image_name, n):
                continue

            # Executa o algoritmo e calcula o tempo de execução
            start = time.time()
            new_image = kmeans(path, n)
            timer = time.time() - start

            # Salva a imagem e faz o log final da execução
            output = simage(new_image, image_name, n, output_id)
            print_info(output, timer=timer, original_size=original_size)
            mimagem(new_image)

