map_info = ""
map_images = []

def update_map_info(info):
    global map_info
    global map_images
    map_images = info.pop("images")
    map_info = info.pop("info")

def get_map_info():
    return map_info

def get_map_images():
    return map_images
