import hash_function
from extract import extract
from load_process import load_process

class Table:
    def __init__(self, hash_size, dim):
        self.table = {}
        self.hash_size = hash_size

    def add(self, id, label, hash):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}

        # Add the hash values to the current table.
        if hash in self.table:
            self.table[hash].append(entry)
        else:
            self.table[hash] = [entry]

    def query(self, vectors):
        # Compute hash value for the query vector.
        hashes = hash_function.hash_func(vectors)
        results = []
        # Loop over the query hashes and determine if they exist in
        # the current table.
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results


class LSH:
    def __init__(self, hash_size, dim, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size, dim))

    def add(self, id, label, h):
        for table in self.tables:
            table.add(id, label, h)

    def query(self, vectors):
        results = []
        for table in self.tables:
            results.extend(table.query(vectors))
        return results


class BuildLSHTable:
    def __init__(
            self,
            hash_size=8,
            dim=2048,
            num_tables=10,
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.dim, self.num_tables)

    def query(self, path, verbose=True):
        # Compute the embeddings of the query image and fetch the results.
        # from keras.applications.resnet50 import preprocess_input
        # from keras.preprocessing import image
        # img = image.load_img(path, target_size=(224, 224))
        xs = load_process(path)
        features = extract(xs)

        results = self.lsh.query(features)
        if verbose:
            print("Matches:", len(results))

        # Calculate Jaccard index to quantify the similarity.
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.dim
        return counts


lsh_builder = BuildLSHTable()


def check_tablle():
    idx = 0
    for hash, entry in lsh_builder.lsh.tables[0].table.items():
        if idx == 5:
            break
        if len(entry) < 5:
            print(hash, entry)
            idx += 1
