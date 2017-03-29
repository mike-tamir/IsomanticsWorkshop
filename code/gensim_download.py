from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import pickle
import zipfile


def googauth():
    """Google authentication"""
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("../gauth.yml")
    # if gauth.credentials is None:
    #     gauth.LocalWebserverAuth()
    # elif gauth.access_token_expired:
    #     gauth.Refresh()
    # else:
    #     gauth.Authorize()

    # gauth.SaveCredentialsFile("../gauth.yml")
    gauth.Authorize()
    return gauth


def retrieve_files(drive, ids, path, filenames):
    """Retrieve files from google drive
    drive = GoogleDrive() Object
    ids = list of google drive ids to download
    path = local path destination
    filenames = list of filenames for local save"""

    for i, file in enumerate(ids):
        f = drive.CreateFile({'id': file})
        f.GetContentFile(path + filenames[i])


def unzip_files(path, filenames):
    for file in filenames:
        zipfile.ZipFile(path + file + '.zip').extractall(path + file)
        os.remove(path + file + '.zip')


def pickle_rw(loc, *tuples, write=True):
    """Pickle object in each tuple to/from location folder
    loc = folder location for objects
    tuples = the filenames and objects to pickle ('name', name)"""
    loc = loc if loc[-1] == '/' else loc + '/'
    result = []
    for tup in tuples:
        fname, obj = tup
        if write:
            with open(loc + fname + '.pkl', 'wb') as f:
                pickle.dump(obj, f)
        else:
            with open(loc + fname + '.pkl', 'rb') as f:
                result.append(pickle.load(f))
    if result == []:
        return
    else:
        return result


if __name__ == "__main__":
    # Set google drive file ids and associated languages
    gensim_fileids = """0B0ZXk88koS2KX01rR2dyRWpHNTA
0B0ZXk88koS2KYkd5OVExR3o1V1k
0B0ZXk88koS2KNER5UHNDY19pbzQ
0B0ZXk88koS2KcW1aTGloZnpCMGM
0B0ZXk88koS2KQnNvcm9UUUxPVXc
0B0ZXk88koS2KblhZYmdReE9vMXM
0B0ZXk88koS2KVnFyem4yQkxJUFk
0B0ZXk88koS2KM0pVTktxdG15TkE
0B0ZXk88koS2KLVVLRWt0a3VmbDg
0B0ZXk88koS2KZkhLLXJvbXVhbzQ
0B0ZXk88koS2KX2xLamRlRDJ3N1U
0B0ZXk88koS2KQWxEemNNUHhnTWc
0B0ZXk88koS2KTlM3Qm1Ta2FBaTg
0B0ZXk88koS2KVVNDS0lqdGNOSGM
0B0ZXk88koS2KMzRjbnE4ZHJmcWM
0B0ZXk88koS2KbDhXdWg1Q2RydlU
0B0ZXk88koS2KelpKdHktXzlNQzQ
0B0ZXk88koS2KOEZ4OThyS3gxZHM
0B0ZXk88koS2KOWdOYk5KaVhrX2c
0B0ZXk88koS2KbFlmMy1PUHBSZ0E
0B0ZXk88koS2KRDcwcV9IVWFTeUE
0B0ZXk88koS2KMUJxZ0w0WjRGdnc
0B0ZXk88koS2KNGNrTE4tVXRUZFU
0B0ZXk88koS2Kcl90XzBYZ0lxMkE
0B0ZXk88koS2KNk1odTJtNkUxcEk
0B0ZXk88koS2KajRzX2VuYkVtYzQ
0B0ZXk88koS2KV1FJN0xRX1FxaFE
0B0ZXk88koS2KVDNLallXdlVQbUE
0B0ZXk88koS2KUHZZZkVwd1RoVmc"""
    gensim_fileids = gensim_fileids.split('\n')

    gensim_languages = ['Bengali',
                        'Catalan',
                        'Chinese',
                        'Danish',
                        'Dutch',
                        'Esperanto',
                        'Finnish',
                        'French',
                        'German',
                        'Hindi',
                        'Hungarian',
                        'Indonesian',
                        'Italian',
                        'Japanese',
                        'Javanese',
                        'Korean',
                        'Malay',
                        'Norwegian',
                        'Norwegian Nynorsk',
                        'Polish',
                        'Portuguese',
                        'Russian',
                        'Spanish',
                        'Swahili',
                        'Swedish',
                        'Tagalog',
                        'Thai',
                        'Turkish',
                        'Vietnamese']

    gensim_lgs = ['bn',
                  'ca',
                  'zh',
                  'da',
                  'nl',
                  'eo',
                  'fi',
                  'fr',
                  'de',
                  'hi',
                  'hu',
                  'id',
                  'it',
                  'ja',
                  'jv',
                  'ko',
                  'ms',
                  'no',
                  'nn',
                  'pl',
                  'pt',
                  'ru',
                  'es',
                  'sw',
                  'sv',
                  'tl',
                  'th',
                  'tr',
                  'vi']

    # Set google Auth and instantiate drive
    gauth = googauth()
    drive = GoogleDrive(gauth)

    # Download all file ids from google
    retrieve_files(drive, gensim_fileids, 'data/gensim/',
                   [lg + '.zip' for lg in gensim_lgs])

    # Unzip all files
    unzip_files('data/gensim/', gensim_lgs)

    # Pickle objects for later
    pickle_rw('pickle/', ('gensim_languages', gensim_languages),
              ('gensim_lgs', gensim_lgs))
