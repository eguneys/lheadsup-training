import itertools
import struct
import gzip
import sys
import numpy as np

V6_STRUCT_STRING = '9s13s10s'

class ChunkParser:
    def __init__(self, 
            chunks,
            sample=1,
            batch_size=256):
        self.inner = ChunkParserInner(self, chunks, sample, batch_size)

    def parse(self):
        return self.inner.parse()

    def shutdown(self):
        pass

class ChunkParserInner:

    def __init__(self, parent, chunks, sample, batch_size):

        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                    (np.zeros(8, dtype=np.float32) + i).tobytes())

        self.sample = sample


        self.chunks = chunks
        self.batch_size = batch_size

        self.init_structs()

    def init_structs(self):
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)


    def parse(self):
        gen = self.v6_gen(self.chunks)
        gen = self.tuple_gen(gen)
        gen = self.batch_gen(gen)

        for b in gen:
            yield b

    def batch_gen(self, gen):
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return
            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]),
                    b''.join([x[2] for x in s]))

    def tuple_gen(self, gen):
        for r in gen:
            yield self.convert_v6_to_tuple(r)

    def task(self, filename):
        for item in self.single_file_gen(filename):
            yield item

    def v6_gen(self, chunk_filenames):
        while len(chunk_filenames):
            filename = chunk_filenames.pop()
            for item in self.task(filename):
                yield item


    def convert_v6_to_tuple(self, content):
        (prob_ranks, prob_cards, planes) = self.v6_struct.unpack(content)

        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)

        planes = planes.tobytes() + \
                self.flat_planes[1]
        assert len(planes) == ((5 * 2 + 1) * 8 * 4)

        return (planes, prob_ranks, prob_cards)

    def single_file_gen(self, filename):
        try:
            record_size = self.v6_struct.size
            with gzip.open(filename, 'rb') as chunk_file:
                while True:
                    chunkdata = chunk_file.read(256 * record_size)
                    if len(chunkdata) == 0:
                        break
                    for item in self.sample_record(chunkdata):
                        yield item

        except:
            print("failed to parse {}".format(filename))
            sys.exit(1)

    def sample_record(self, chunkdata):
        record_size = self.v6_struct.size

        for i in range(0, len(chunkdata), record_size):
            if self.sample > 1:
                if random.randint(0, self.sample - 1) != 0:
                    continue
            record = chunkdata[i:i+record_size]
            yield record


