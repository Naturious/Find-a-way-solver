import os  # For command line argument management
import sys
import json
import cv2
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')


def check_local():
    """Returns true if running on local environment

    :return: Indicator if the runtime is an AWS Lambda runtime
    :rtype: bool
    """
    return 'LAMBDA_TASK_ROOT' not in os.environ


def save_image(filename, img, bucket = None):
    """

    :param bucket:
    :param filename: path to the original image
    :param img: cv2 image to be saved
    :return: path to new saved image
    """

    if bucket is not None:
        # Upload the file
        print(f"Writing result locally to {filename}")
        cv2.imwrite(filename, img)
        try:
            key = "solved/" + os.path.basename(filename)
            print(f"Uploading into bucket {bucket} with key {key}")
            response = s3_client.upload_file(filename, bucket, key)
        except ClientError as e:
            print(e)
            return False
        print(f"Successfully uploaded result image to {bucket}")
        return True
    else:
        saved_path = os.path.join(os.path.dirname(filename), "solved_" + os.path.basename(filename))
        cv2.imwrite(saved_path, img)
        print(f"Saved image into {saved_path}")

def image_to_board(img):
    # Get point contours in a mask:
    global line_color, second_box, min_box
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(grayscaled, (0, 0), (700, 90), (255, 255, 255), -1)
    cv2.rectangle(grayscaled, (0, 745), (700, 800), (255, 255, 255), -1)
    full_mask = cv2.bitwise_not(
        cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1))

    # Find first point and calculate centroid then calculate all other
    # centers out of it, assuming we know the size from the beginning
    contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find top left point:
    """Method description:
    Find top-left corner position (min_x,min_y) 
    and second minimum x by going through the bounding boxes x and y,
    comparing max and min values to the current box values
    """
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    first_box = cv2.boundingRect(contours[0])
    min_x = first_box[0]
    min_y = first_box[1]
    second_x = 99999

    epsilon = 3  # This is because black box bounding rects and dot bounding rects are off by about 1.5 pixels
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if x <= min_x + epsilon and y <= min_y + epsilon:
            min_x = x
            min_y = y
            min_box = i
        elif x < second_x and x - min_x > epsilon:
            second_x = x
            second_box = i

    m = cv2.moments(contours[min_box])
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])

    m2 = cv2.moments(contours[second_box])
    d = m2["m10"] / m2["m00"] - m["m10"] / m["m00"]

    print("Min point: ({0},{1}): d = {2}".format(min_x, min_y, d))

    # Find size
    size = [0, 0]  # (hori,vert)
    i = c_x
    while sum(img[c_y][int(i)]) != 765:  # An efficient way to check if the color is white (765 = 255*3)
        size[0] += 1
        i += d
    i = c_y
    while sum(img[int(i)][c_x]) != 765:
        size[1] += 1
        i += d

    print("Puzzle of size: " + str(size[0]) + "x" + str(size[1]))

    puzzle = [[0 for i in range(size[1])] for j in range(size[0])]

    for x in range(size[0]):
        for y in range(size[1]):
            current_pixel = (c_x + int(x * d), c_y + int(y * d))
            color_sum = sum(img[current_pixel[1]][current_pixel[0]])
            if color_sum == 612:  # Regular point
                puzzle[x][y] = 1
            elif color_sum == 420:  # Starting point: red
                puzzle[x][y] = 2
                line_color = (111, 76, 233)
            elif color_sum == 355:  # Starting point: green
                puzzle[x][y] = 2
                line_color = (136, 192, 27)
            elif color_sum == 0:
                pass
            else:
                print("Found unexpected color at pixel ({0},{1})".format(x, y))

    top_left = (c_x, c_y)
    return puzzle, top_left, line_color, d  # puzzle matrix, top left corner coordinates, distance between points


def print_puzzle(puzzle):
    trans = [list(x) for x in zip(*puzzle)]
    for x in range(len(trans)):
        print(trans[x])


# Representing the graph as list of adjacency lists, accessing nodes with their X and Y coordinates

class Graph:
    puzz = 0
    adj_list = {}
    nodes = []
    starting_point = 0

    def __init__(self, puzzle):
        self.puzz = puzzle
        for x in range(len(puzzle)):
            for y in range(len(puzzle[0])):
                if puzzle[x][y] == 2:
                    self.starting_point = (x, y)
                if puzzle[x][y]:
                    self.add_node((x, y))
                    # Right
                    if self.valid_coordinates(x + 1, y) and puzzle[x + 1][y]:
                        self.add_neighbour((x, y), (x + 1, y))
                    # Left
                    if self.valid_coordinates(x - 1, y) and puzzle[x - 1][y]:
                        self.add_neighbour((x, y), (x - 1, y))
                    # Up
                    if self.valid_coordinates(x, y - 1) and puzzle[x][y - 1]:
                        self.add_neighbour((x, y), (x, y - 1))
                    # Down
                    if self.valid_coordinates(x, y + 1) and puzzle[x][y + 1]:
                        self.add_neighbour((x, y), (x, y + 1))

    def add_node(self, coordinates):
        self.nodes.append(coordinates)
        self.adj_list[coordinates] = []

    def add_neighbour(self, node1, node2):
        self.adj_list[node1].append(node2)

    def is_neighbour(self, node1, node2):
        return node2 in self.adj_list[node1]

    def valid_coordinates(self, x, y):
        return 0 <= x < len(self.puzz) and 0 <= y < len(self.puzz[0])

    def get_nodes(self):
        return self.nodes

    def size(self):
        return len(self.nodes)

    def get_node_neighbours(self, coordinates):
        return self.adj_list[coordinates]

    def __str__(self):
        return str(self.adj_list)

    def print(self):
        for node in self.adj_list:
            print(node, self.adj_list[node])


def find_path(graph, visited=None):
    """ The problem comes down to solving the hamiltonian path problem
    starting at the starting point.
    I'll be using a DFS approach described in here:
    https://www.hackerearth.com/practice/algorithms/graphs/hamiltonian-path/
    In short, it consists of multiple DFSs starting at the
    starting vertex to see if one of the searches gets through all the
    nodes (that will be checked by checking if the stack size equals the graph size)
    ** This approach can and will (if I find time) upgrade to a dp approach which is
    far more scalable and elegant
    """

    # Brute force approach:
    if len(visited) == graph.size():
        # We found a path that goes through all the nodes
        return True, visited

    for node in graph.adj_list[visited[-1]]:
        if node not in visited:
            visited.append(node)
            if find_path(graph, visited)[0]:
                return True, visited
            visited.pop()
    return False, visited


def draw_path(path, img, pt, line_color, d):
    """
        Draws "path" to "img" having "pt" as
        top left corner coordinates and d as
        distance between nodes
    """
    for i in range(len(path) - 1):
        center1 = (pt[0] + int(path[i][0] * d), pt[1] + int(path[i][1] * d))
        center2 = (pt[0] + int(path[i + 1][0] * d), pt[1] + int(path[i + 1][1] * d))
        cv2.line(img, center1, center2, line_color, 7)


def get_filename(event):
    # If no filename is specified in the event, get it from argv
    if event is None:
        if len(sys.argv) != 2:
            print("Usage: python3 finder.py [screenshot]")
            sys.exit(0)
        if not (os.path.isfile(sys.argv[1])):
            print("'%s' is not a file or can't be opened" % sys.argv[1])
            sys.exit(0)
        return sys.argv[1], None
    else:
        # This will pull data from S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        print(f"Bucket: {bucket}, Key: {key}")
        filename = os.path.abspath(os.path.join(os.sep, 'tmp', bucket+"-"+key.replace("/","-")))
        print(f"Filename to download file to: {filename}")
        with open(filename, 'wb') as f:
            s3_client.download_fileobj(bucket, key, f)
            print(f"Successfully downloaded S3 file into {filename}")
        return filename, bucket


##########################################################################

def lambda_handler(event=None, context=None):
    # Check if I'm running locally or on a Lambda runtime environment
    if check_local():
        print("Running locally")
    else:
        print("Running on an AWS Lambda runtime environment")

    filename, bucket = get_filename(event)
    size = os.path.getsize(filename)
    print(f"Filename {filename} with size {size}")
    if size == 0:
        print("File is empty, nothing to process")
        return {
            "statusCode":200,
            "body":json.dumps("Empty file, nothing to process")
        }

    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    ## Parsing the image to an array along with the
    # coordinates of the top left corner and the distance between points
    puzzle, top_left, line_color, distance = image_to_board(img)

    print_puzzle(puzzle)

    ## Convert array to graph
    graph = Graph(puzzle)

    ## Find path

    res, ham_path = find_path(graph, [graph.starting_point])  # Returns the path into ham_path by use of the "global" keyword
    if res:
        print("Found a hamiltonian path")
    else:
        print("Didnt find a hamiltonian path")
    draw_path(ham_path, img, top_left, line_color, distance)
    # Save image into a file
    save_image(filename, img, bucket)

    return {
        "statusCode":200,
        "body":json.dumps("Successfully executed lambda")
    }