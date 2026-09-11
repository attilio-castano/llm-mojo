"""Exact chat framing and state transitions; no model weights required."""
from std.testing import assert_equal, assert_raises
from llm_mojo.chat import ChatHistory, DEFAULT_SYSTEM, IM_END, NEWLINE
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace, TableReader
from test_tokenizer import equal_ids, fixture_bytes


def main() raises:
    var tokenizer = Tokenizer("build/checkpoints/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775/prepared-v1/tables.bin")
    var work = TokenizerWorkspace()
    equal_ids(tokenizer.encode("\n",work),[NEWLINE])
    equal_ids(tokenizer.encode("<|im_end|>",work),[IM_END])
    var reader = TableReader("build/oracle_data/chat.bin")
    var count = reader.ints()[0]
    var checks = 0
    for _ in range(count):
        var system = String(from_utf8=fixture_bytes(reader))
        var history = ChatHistory(tokenizer,work,system)
        var turns = reader.ints()[0]
        for _ in range(turns):
            var user = String(from_utf8=fixture_bytes(reader))
            var expected = reader.ints()
            var reply = reader.ints()
            history.begin(tokenizer,work,user,256,4096)
            equal_ids(history.tokens,expected)
            for token in reply:
                history.accept(token)
            history.accept(IM_END)
            assert_equal(history.generating,False)
            assert_equal(history.reason,"stop")
            checks += 1
    assert_equal(reader.pos,len(reader.data))
    var h = ChatHistory(tokenizer,work,String(DEFAULT_SYSTEM))
    var original = h.tokens.copy()
    with assert_raises():
        h.begin(tokenizer,work,"hi",4096,4096)
    equal_ids(h.tokens,original)
    h.begin(tokenizer,work,"hi",1,4096)
    h.accept(42)
    assert_equal(h.reason,"limit")
    assert_equal(h.tokens[len(h.tokens)-3],42)
    assert_equal(h.tokens[len(h.tokens)-2],IM_END)
    assert_equal(h.tokens[len(h.tokens)-1],NEWLINE)
    var completed = h.tokens.copy()
    h.finish("interrupted")
    equal_ids(h.tokens,completed)
    h.reset()
    equal_ids(h.tokens,original)
    h.begin(tokenizer,work,"hi",2,4096)
    h.finish("interrupted")
    assert_equal(h.generated,0)
    assert_equal(h.reason,"interrupted")
    assert_equal(h.tokens[len(h.tokens)-2],IM_END)
    h.begin(tokenizer,work,"continue",2,4096)
    h.accept(151643)
    assert_equal(h.reason,"stop")
    assert_equal(h.tokens[len(h.tokens)-3],151643)
    assert_equal(h.tokens[len(h.tokens)-2],IM_END)
    with assert_raises():
        h.accept(42)
    h.reset()
    h.begin(tokenizer,work,"hi",2,4096)
    var size = len(h.tokens)
    h.finish("interrupted")
    h.reset()
    # Exact capacity boundary reserves reply plus both closure tokens.
    h.begin(tokenizer,work,"hi",2,size+4)
    h.reset()
    with assert_raises():
        h.begin(tokenizer,work,"hi",2,size+3)
    equal_ids(h.tokens,original)
    print("chat framing passed:",checks,"HF prefixes; limits stop interruption reset atomic rejection")
