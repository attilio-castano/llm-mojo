"""Standalone native tokenizer; setup and artifact verification precede launch."""
from std.sys import argv
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: tokenizer tables.bin encode|decode value skip-special"
        )
    var tokenizer = Tokenizer(args[1])
    if args[2] == "encode" or args[2] == "encode-file":
        var workspace = TokenizerWorkspace()
        var ids = List[Int]()
        if args[2] == "encode-file":
            var bytes = open(args[3], "r").read_bytes()
            ids = tokenizer.encode_bytes(bytes, workspace)
        else:
            ids = tokenizer.encode(args[3], workspace)
        var result = String("[")
        for i in range(len(ids)):
            if i > 0:
                result += ", "
            result += String(ids[i])
        result += "]"
        print(result)
    elif args[2] == "decode":
        var ids = List[Int]()
        if args[3].byte_length() > 0:
            for piece in args[3].split(","):
                ids.append(Int(String(piece)))
        print(tokenizer.decode(ids, args[4] == "1"))
    else:
        raise Error("unsupported tokenizer operation")
