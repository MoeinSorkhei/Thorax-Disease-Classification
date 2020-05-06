import urllib.request
import os
import tarfile

import helper


def download_data(params):
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    download_path = params['download_path']
    helper.make_dir_if_not_exists(download_path)

    for idx, link in enumerate(links):
        file_name = download_path + '/' + 'images_%02d.tar.gz' % idx

        print(f'In [download_data]: downloading at "{file_name}"...')
        urllib.request.urlretrieve(link, file_name)  # download the zip file
    print("In [download_data]: download complete. \n")


def extract_data(params):
    archive_path, extract_path = params['download_path'], params['extracted_path']
    helper.make_dir_if_not_exists(extract_path)

    for fname in os.listdir(archive_path):
        file_path = f'{archive_path}/{fname}'

        tar = tarfile.open(file_path, 'r:gz')
        tar.extractall(path=extract_path)

        print(f'In [extract_data]: extracted "{archive_path}/{fname}" to "{extract_path}".')
        tar.close()
    print('In [extract_data]: all done.')


def prepare_data(params):
    # adding ../ because data should be in the project folder while the code is running from the src folder
    # params['download_path'] = '../' + params['download_path']
    # params['extracted_path'] = '../' + params['extracted_path']
    download_data(params)
    extract_data(params)
