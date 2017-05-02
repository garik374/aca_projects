import urllib

# MNIST data url
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def get_data(file_name, file_path):
    urllib.request.urlretrieve(SOURCE_URL + file_name, file_path)
    return file_path
