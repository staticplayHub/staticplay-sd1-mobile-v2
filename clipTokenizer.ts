export type ClipTokenizerFiles = {
  vocabJson: string;
  mergesTxt: string;
};

type Vocab = Record<string, number>;

function bytesToUnicode() {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const byteToUnicode = new Map<number, string>();
  const unicodeToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i]);
    byteToUnicode.set(bs[i], ch);
    unicodeToByte.set(ch, bs[i]);
  }
  return { byteToUnicode, unicodeToByte };
}

function getPairs(word: string[]) {
  const pairs = new Set<string>();
  let prev = word[0];
  for (let i = 1; i < word.length; i++) {
    const curr = word[i];
    pairs.add(`${prev}\u0001${curr}`);
    prev = curr;
  }
  return pairs;
}

function bpe(token: string, bpeRanks: Map<string, number>) {
  let word = token.split('');
  if (word.length === 1) return token;

  while (true) {
    const pairs = getPairs(word);
    if (pairs.size === 0) break;

    let minPair: string | null = null;
    let minRank = Infinity;
    for (const p of pairs) {
      const r = bpeRanks.get(p);
      if (r != null && r < minRank) {
        minRank = r;
        minPair = p;
      }
    }
    if (minPair == null) break;
    const [first, second] = minPair.split('\u0001');

    const newWord: string[] = [];
    let i = 0;
    while (i < word.length) {
      const j = word.indexOf(first, i);
      if (j === -1) {
        newWord.push(...word.slice(i));
        break;
      }
      newWord.push(...word.slice(i, j));
      i = j;
      if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
        newWord.push(first + second);
        i += 2;
      } else {
        newWord.push(word[i]);
        i += 1;
      }
    }
    word = newWord;
    if (word.length === 1) break;
  }
  return word.join(' ');
}

function utf8ToBytes(text: string): Uint8Array {
  // Hermes usually provides TextEncoder, but keep a fallback.
  const te = (globalThis as any).TextEncoder ? new (globalThis as any).TextEncoder() : null;
  if (te) return te.encode(text);

  const out: number[] = [];
  for (let i = 0; i < text.length; i++) {
    let codePoint = text.charCodeAt(i);
    if (codePoint < 0x80) out.push(codePoint);
    else if (codePoint < 0x800) {
      out.push(0xc0 | (codePoint >> 6));
      out.push(0x80 | (codePoint & 0x3f));
    } else {
      out.push(0xe0 | (codePoint >> 12));
      out.push(0x80 | ((codePoint >> 6) & 0x3f));
      out.push(0x80 | (codePoint & 0x3f));
    }
  }
  return Uint8Array.from(out);
}

export function createClipTokenizer(files: ClipTokenizerFiles) {
  const vocab: Vocab = JSON.parse(files.vocabJson);
  const merges = files.mergesTxt
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'));

  // First merge line is a version header in many BPE files.
  const mergePairs = merges[0].includes('version') ? merges.slice(1) : merges;
  const bpeRanks = new Map<string, number>();
  for (let i = 0; i < mergePairs.length; i++) {
    const parts = mergePairs[i].split(/\s+/);
    if (parts.length !== 2) continue;
    bpeRanks.set(`${parts[0]}\u0001${parts[1]}`, i);
  }

  const { byteToUnicode } = bytesToUnicode();

  const bosToken = '<|startoftext|>';
  const eosToken = '<|endoftext|>';
  const bosId = vocab[bosToken] ?? 49406;
  const eosId = vocab[eosToken] ?? 49407;

  const cache = new Map<string, number[]>();

  // ASCII-ish regex (works on Hermes without Unicode property escapes).
  const pat =
    /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\\d+| ?[^\\sA-Za-z\\d]+|\\s+(?!\\S)|\\s+/g;

  function encode(text: string, maxLen = 77) {
    const cached = cache.get(`${text}\u0001${maxLen}`);
    if (cached) return cached.slice();

    const matches = text.match(pat) ?? [];
    const bpeTokens: number[] = [];

    for (const m of matches) {
      const bytes = utf8ToBytes(m);
      let chars = '';
      for (const b of bytes) chars += byteToUnicode.get(b) ?? '';

      const bpeOut = bpe(chars, bpeRanks);
      for (const tok of bpeOut.split(' ')) {
        const id = vocab[tok];
        bpeTokens.push(id == null ? eosId : id);
      }
    }

    const ids: number[] = [bosId, ...bpeTokens, eosId];
    if (ids.length > maxLen) ids.length = maxLen;
    while (ids.length < maxLen) ids.push(eosId);

    cache.set(`${text}\u0001${maxLen}`, ids.slice());
    return ids;
  }

  return { encode, bosId, eosId };
}

