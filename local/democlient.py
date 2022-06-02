import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 11234))
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

msg = s.recv(1024)
print(msg.decode("utf-8"))

s.sendall(bytes("It is test test test from client.", 'utf-8'))
#s.send(b'\x00\x01\x02\x03\x00\x11\x12\x13\x20\x21\x22\x23\x30\x31\x32\x33')

time.sleep(3)
s.sendall(bytes("It is second message.", 'utf-8'))

time.sleep(2)
s.sendall(bytes("It is third message.", 'utf-8'))
