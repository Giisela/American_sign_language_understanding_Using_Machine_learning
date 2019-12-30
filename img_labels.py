import os

def labels_list(images_dir_path):
    """
    Itera recursivamente pelo diretorio para listar todas as informações de todas as imagens encontradas
    Retorna a lista dos paths das imagens
    """
    images_labels_list = []
    print('Images directory - "{}"'.format(images_dir_path))
    for (dirpath, dirnames, filenames) in os.walk(images_dir_path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            label = os.path.splitext(os.path.basename(dirpath))[0]
            info = {}
            info['image_path'] = path
            info['image_label'] = label
            images_labels_list.append(info)
    return images_labels_list


def write_labels_to_file(images_labels_list, output_file_path):
    """
    Escreve a lista das labels das imagens num ficheiro
    """

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        for info in images_labels_list:
            path = info['image_path']
            label = info['image_label']
            line = path + "\t" + label + '\n'
            output_file.write(line)



def init_img_labels(images_source):

    if images_source not in ['train', 'test']:
        print("Invalid image-source '{}'!".format(images_source))
        return
    images_dir_path = ('./dados/images/{}'.format(images_source))
    images_labels_path = ('./dados/labels/{}ing_images_labels.txt'.format(images_source))

    print("Gathering info about images at path '{}'...".format(images_dir_path))
    images_labels_list = labels_list(images_dir_path)
    print("Done!")

    print("Writing images labels info to file at path '{}'...".format(images_labels_path))

    write_labels_to_file(images_labels_list, images_labels_path)
    print("Done!")


