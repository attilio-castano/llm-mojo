"""Darwin terminal input and synchronous Ctrl-C handling, without signal handlers.

Block SIGINT before GPU threads are created. Consume pending SIGINT between
model calls; never run allocation, GPU work or language code in a handler.
"""
from std.ffi import external_call, c_ssize_t
from std.memory import Pointer

@fieldwise_init
struct PollFD(Copyable, Movable):
    var fd: Int32
    var events: Int16
    var revents: Int16


def block_interrupt() raises:
    var mask = UInt32(2)  # Darwin sigset_t: bit SIGINT-1.
    var previous = UInt32(0)
    if external_call["sigprocmask",Int32](Int32(1),Pointer(to=mask),Pointer(to=previous)) != 0:
        raise Error("could not block SIGINT")


def interrupted() raises -> Bool:
    var pending = UInt32(0)
    if external_call["sigpending",Int32](Pointer(to=pending)) != 0:
        raise Error("could not inspect SIGINT")
    if pending & UInt32(2) == 0:
        return False
    var mask = UInt32(2)
    var caught = Int32(0)
    if external_call["sigwait",Int32](Pointer(to=mask),Pointer(to=caught)) != 0:
        raise Error("could not consume SIGINT")
    return True


def read_line(mut bytes: List[UInt8]) raises -> Int:
    """Return 1 for a line, 0 for EOF, -1 for Ctrl-C, -2 for an oversized line.

    Read one byte at a time so piped future turns remain unread during generation.
    Terminal canonical editing and echo remain under the OS's control.
    """
    bytes.clear()
    var oversized = False
    while True:
        if interrupted():
            bytes.clear()
            return -1
        var descriptor = PollFD(0,1,0)  # stdin, POLLIN
        var ready = external_call["poll",Int32](Pointer(to=descriptor),UInt32(1),Int32(50))
        if ready < 0:
            raise Error("terminal poll failed")
        if ready == 0:
            continue
        var byte = UInt8(0)
        var count = external_call["read",c_ssize_t](Int(0),Pointer(to=byte),Int(1))
        if count < 0:
            raise Error("terminal read failed")
        if count == 0:
            return -2 if oversized else (1 if len(bytes)>0 else 0)
        if byte == UInt8(10):
            if len(bytes)>0 and bytes[len(bytes)-1] == UInt8(13):
                _ = bytes.pop()
            return -2 if oversized else 1
        if len(bytes) < 65536:
            bytes.append(byte)
        else:
            oversized = True
