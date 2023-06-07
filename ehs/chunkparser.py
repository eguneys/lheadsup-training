import itertools
import struct
import gzip
import sys
import numpy as np
import random
import shufflebuffer as sb
import multiprocessing as mp
import os

V6_STRUCT_STRING = '>14sf'

def chunk_reader(chunk_filenames, chunk_filename_queue):
    chunks = []
    done = chunk_filenames
    while True:
        if not chunks:
            chunks, done = done, chunks
            random.shuffle(chunks)
        if not chunks:
            print("chunk_reader didn't find any chunks.")
            return None
        while len(chunks):
            filename = chunks.pop()
            done.append(filename)
            chunk_filename_queue.put(filename)
    print("chunk_reader exiting.")
    return None

class ChunkParser:
    def __init__(self, 
            chunks,
            shuffle_size=1,
            sample=1,
            batch_size=256):
        self.inner = ChunkParserInner(self, chunks, shuffle_size, sample, batch_size)

    def parse(self):
        return self.inner.parse()

    def shutdown(self):
        for i in range(len(self.processes)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.inner.readers[i].close()
            self.inner.writers[i].close()
        self.chunk_process.terminate()
        self.chunk_process.join()

class ChunkParserInner:

    def __init__(self, parent, chunks, shuffle_size, sample, batch_size):


        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                    (np.zeros(8, dtype=np.float32) + i).tobytes())

        self.sample = sample

        self.chunks = chunks
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size

        workers = max(1, mp.cpu_count() - 2)

        if workers > 0:
            print("Using {} worker processes.".format(workers))

            self.readers = []
            self.writers = []
            parent.processes = []
            self.chunk_filename_queue = mp.Queue(maxsize=4096)
            for _ in range(workers):
                read, write = mp.Pipe(duplex=False)
                p = mp.Process(target=self.task,
                        args = (self.chunk_filename_queue, write))

                p.daemon = True
                parent.processes.append(p)
                p.start()
                self.readers.append(read)
                self.writers.append(write)

            parent.chunk_process = mp.Process(target=chunk_reader,
                    args=(chunks,
                        self.chunk_filename_queue))
            parent.chunk_process.daemon = True
            parent.chunk_process.start()
        else:
            self.chunks = chunks

        self.init_structs()

    def init_structs(self):
        self.v6_struct = struct.Struct(V6_STRUCT_STRING)


    def parse(self):
        gen = self.v6_gen()
        gen = self.tuple_gen(gen)
        gen = self.batch_gen(gen)
        for b in gen:
            yield b

    def batch_gen(self, gen):
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return

            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]))

    def tuple_gen(self, gen):
        for r in gen:
            yield self.convert_v6_to_tuple(r)

    def task(self, chunk_filename_queue, writer):
        self.init_structs()

        while True:
            filename = chunk_filename_queue.get()
            for item in self.single_file_gen(filename):
                writer.send_bytes(item)
                

    def v6_gen(self):
        sbuff = sb.ShuffleBuffer(self.v6_struct.size, self.shuffle_size)
        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def convert_v6_to_tuple(self, content):
        (cards, value) = self.v6_struct.unpack(content)

        value = struct.pack('f', value)

        planes = np.unpackbits(np.frombuffer(cards, dtype=np.uint8)).astype(np.float32)

        planes = planes[0:4*8].tobytes() + \
                self.flat_planes[1] + \
                planes[4*8:].tobytes() + \
                self.flat_planes[1]

        assert len(planes) == ((7 * 2 + 1 + 1) * 8 * 4)

        return (planes, value)

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


