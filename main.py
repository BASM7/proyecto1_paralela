import sys
import getopt
from worm import Worm
from mpi4py import MPI


def load_data(filename):
    loaded_data = []
    f = open(filename, "r")
    filas = f.readlines()
    for fila in filas:
        tlist = [int(s) for s in fila.__str__().split(',')]
        tlist = tlist[0:-1]
        loaded_data.append(tlist)
    return loaded_data


def get_command_line_values(argv):
    r_str = ""
    g_str = ""
    s_str = ""
    i_str = ""
    l_str = ""

    try:
        opts, args = getopt.getopt(argv, "r:g:s:i:l", ["R=", "G=", "S=", "I=", "L="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-r", "--R"):
            r_str = arg
        elif opt in ("-g", "--G"):
            g_str = arg
        elif opt in ("-s", "--S"):
            s_str = arg
        elif opt in ("-i", "--I"):
            i_str = arg
        elif opt in ("-l", "--L"):
            l_str = arg

    return int(r_str), int(g_str), int(s_str), int(i_str), int(l_str)


def main():
    list_data = load_data("poker-hand-training-true.data")
    for plist in list_data:
        print(plist)


if __name__ == '__main__':
    main()

