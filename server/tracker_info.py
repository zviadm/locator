import copy
map_info = ""
map_images = []
device_stats = {}

def update_map_info(info):
    global map_info
    global map_images
    global device_stats
    map_images = info.pop("images")
    map_info = info.pop("info")

    device_stats_cpy = copy.deepcopy(device_stats)
    device_stats_cpy.update(info.pop("device_stats"))
    device_stats = device_stats_cpy

def get_map_info():
    return map_info

def get_map_images():
    return map_images

def get_device_stats():
    return device_stats
