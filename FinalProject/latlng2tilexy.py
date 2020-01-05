def coordinate_to_tile_index(lng, lat, zoom):
    import math
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def main():
    import sys
    import os

    args_len = len(sys.argv)

    if args_len == 1:
        """ input from stdin (linux pipe)
            like: 105.83293,21.02079,105.83293,21.02079
        """
        data = sys.stdin.read()
        params = data.split(",")
        if len(params) > 2:
            params = params[2:]
            print(params)
        zoom = 18
        lng = float(params[0].strip())
        lat = float(params[1].strip())

    elif args_len > 2:
        """ input from arguments with sperate value
        """
        lat = float(sys.argv[1])
        lng = float(sys.argv[2])
        zoom = int(sys.argv[3])
    
    else:
        """ input from arguments with string like
            105.83293,21.02079,105.83293,21.02079
        """
        data = sys.argv[1]
        params = data.split(",")
        if len(params) > 2:
            params = params[2:]
            print(params)
        zoom = 18
        lng = float(params[0].strip())
        lat = float(params[1].strip())

    (x, y) = coordinate_to_tile_index(lng, lat, zoom)
    print("Lng={} Lat={} Z={}: ".format(lng, lat, zoom), x, y)

    # call feh to view tile
    url = "https://27.72.89.107:8099/i/i/tile/fire/{}/tile_{}_{}_{}.png".format(zoom, x, y, zoom)

    SAVE_PATH = "/tmp/tile_tmp.png"
    os.system("curl -k -o '{}' {}".format(SAVE_PATH, url))
    os.system('feh {}'.format(SAVE_PATH))

if __name__ == "__main__":
    main()