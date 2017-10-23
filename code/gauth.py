from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def update_gauth():
	"""Update gauth, save credentials to yml"""
	gauth = GoogleAuth()
	gauth.LoadClientConfigFile('../client_secrets.json')
	gauth.LocalWebserverAuth()
	gauth.SaveCredentialsFile("../gauth.yml")


if __name__ == "__main__":
	update_gauth()
