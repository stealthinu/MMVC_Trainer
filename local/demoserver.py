import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 11234))
s.listen(5)

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")
    clientsocket.send(bytes("Connected to server completion!", 'utf-8'))

    msg = b''
    while True:
        try:
            msg = clientsocket.recv(256)
            if len(msg) > 0:
                print(msg.decode("utf-8"))
        except ConnectionResetError:
            print("Raise ConnectReset!")
            break

    clientsocket.close()
