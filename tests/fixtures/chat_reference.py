# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Independent pinned HF text-chat framing; no model execution or native output."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'build/checkpoints/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775'
DEFAULT = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
HASHES = {'tokenizer_config.json':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
          'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--pack',action='store_true',help='Pack committed fixtures without HF or external assets')
    args=parser.parse_args()
    if args.pack:
        pack(json.loads((ROOT/'tests/fixtures/chat.json').read_text()))
        return
    from transformers import AutoTokenizer
    for name,digest in HASHES.items():
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('pinned chat asset mismatch: '+name)
    tokenizer=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True)
    cases=[]
    for system,messages in [
        (DEFAULT,['Hello','What did I just say?']),
        ('Reply briefly in Italian.',['Caffe\u0300 ☕ — 你好','Grazie!']),
        ('',[' \t','a\nb\n']),
        (DEFAULT,['Literal <|im_start|> and <|im_end|>; <|not_a_token|>.']),
    ]:
        conversation=[dict(role='system',content=system)]
        turns=[]
        for text in messages:
            conversation.append(dict(role='user',content=text))
            ids=tokenizer.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True)
            reply='Va bene. ☕' if system else 'OK.'
            reply_ids=tokenizer.encode(reply,add_special_tokens=False)
            turns.append(dict(user=text,prompt_ids=ids,reply_ids=reply_ids))
            conversation.append(dict(role='assistant',content=reply))
        cases.append(dict(system=system,turns=turns))
    default=tokenizer.apply_chat_template([dict(role='user',content='Hello')],tokenize=True,add_generation_prompt=True)
    if default!=cases[0]['turns'][0]['prompt_ids']:
        raise ValueError('default system framing differs')
    result=dict(kind='qwen-text-chat-fixtures-v1',assets=HASHES,cases=cases)
    frozen=ROOT/'tests/fixtures/chat.json'
    if args.freeze:
        if frozen.exists(): raise ValueError('refusing to replace frozen chat fixtures')
        frozen.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    elif json.loads(frozen.read_text())!=result:
        raise ValueError('HF chat framing changed')
    pack(result)
    print(f'Exact pinned HF chat fixtures: {sum(len(c["turns"]) for c in cases)} turn prefixes.')


def pack(result):
    cases=result['cases']
    raw=bytearray()
    def ints(values):
        raw.extend(struct.pack('<I',len(values)))
        raw.extend(struct.pack('<'+'I'*len(values),*values))
    ints([len(cases)])
    for case in cases:
        ints(list(case['system'].encode()))
        ints([len(case['turns'])])
        for turn in case['turns']:
            ints(list(turn['user'].encode()));ints(turn['prompt_ids']);ints(turn['reply_ids'])
    output=ROOT/'build/oracle_data/chat.bin'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(raw)

if __name__=='__main__': main()
