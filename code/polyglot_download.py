from pydrive.drive import GoogleDrive
import re
import urllib.request as url
from gensim_download import googauth, pickle_rw


def get_polyglot_links():
    """Get polyglot links from web-site and pickle"""
    address = 'https://sites.google.com/site/rmyeid/projects/polyglot'
    with url.urlopen(address) as f:
        f_str = str(f.read())

    pattern = '<a href="(http://bit.ly/\w{7})" rel="nofollow">'
    pattern += 'polyglot-(\w{2}\w?).pkl</a>'
    polyglot_links = re.findall(pattern, f_str)
    polyglot_links.append(('http://bit.ly/19bTJYC', 'zhc'))

    pickle_rw(('polyglot_links', polyglot_links))
    return polyglot_links


def polyglot_retrieve(polyglot_links):
    """Retrieve files from links and/or google drive"""
    for polyglot_link in polyglot_links:
        address, lg = polyglot_link
        # Retrieve url
        url.urlretrieve(address, '../data/polyglot/' + lg + '.pkl')

        # If the url file was too big, we retrieved an error file.
        # If we obtained the correct file, is is bytes and try will fail.
        try:
            with open('../data/polyglot/' + lg + '.pkl') as f:
                f_str = f.read()
            pattern = '<a href="https://docs.google.com/open\?id=(\w{28})">'
            g_id = re.findall(pattern, f_str)[0]
        except:
            g_id = None

        # If we retrieved the google id from the error file
        if g_id:
            f = drive.CreateFile({'id': g_id})
            f.GetContentFile('../data/polyglot/' + lg + '.pkl')


if __name__ == "__main__":
    # Get polyglot file links from web-site
    polyglot_links = get_polyglot_links()

    # Save polyglot lgs
    polyglot_lgs = [_[1] for _ in polyglot_links]
    pickle_rw(('polyglot_lgs', polyglot_lgs))

    # Set google Auth and instantiate drive
    gauth = googauth()
    drive = GoogleDrive(gauth)

    # Retrive the files
    polyglot_retrieve(polyglot_links)
