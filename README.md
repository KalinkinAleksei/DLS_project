# DLS_project
The reposetory contains results of the final project work for MPTI Deep Learting School 2024 (1 semester)

The aim of the project is to try implement motion blur to a static image. So it is a reverse to image motion debluring task.


# Installation:

Download an `.zip` archive from Google Drive (~2gb):
https://drive.google.com/file/d/1j0lzD40VthaRUp9bANlf8ayFwbv1ELj0/view?usp=sharing

Than run:
```bash
unzip dls_add_motion_blur_project.zip
cd dls_add_motion_blur_project
docker build -t streamlit-app .
```
# Usage:
```bash
docker run -p 8501:8501 streamlit-app
```

