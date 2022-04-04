import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from datetime import datetime
import glob

path = "/Users/kriangsakthuiprakhon/Documents/seniorProject/image/INRIA"

bench = [filename for filename in glob.iglob(path + '**/*.jpg', recursive=True)]
bench = np.sort(np.array(bench))
labels = [filename.split("/")[7] for filename in bench]
train = [[labels[i], bench[i]] for i in range(len(bench))]

large = "/Users/kriangsakthuiprakhon/Documents/seniorProject/image/101_ObjectCategories/"

bench = [filename for filename in glob.iglob(large + '**/*.jpg', recursive=True)]
bench = np.sort(np.array(bench))
labels = [filename.split("/")[7] for filename in bench]
benchmark_data = [[labels[i], bench[i]] for i in range(len(bench))]
# print(labels)
# print(len(train))

def hash_func(embedding, random_vectors):
    embedding = np.array(embedding)
    # Random projection
    bools = np.dot(embedding, random_vectors) > 0
    return bool2int(bools)


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y


class Table:
    def __init__(self, hash_size, dim):
        self.table = {}
        self.hash_size = hash_size
        self.random_vectors = np.random.randn(hash_size, dim).T

    def hashing(self, id, vectors, label):
        # Compute the hash values.
        hashes = hash_func(vectors, self.random_vectors)
        return hashes

    def add(self, id, vectors, label, h):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}
        # Add the hash values to the current table.
        if h in self.table:
            self.table[h].append(entry)
        else:
            self.table[h] = [entry]

    def query(self, vectors):
        # Compute hash value for the query vector.
        hashes = hash_func(vectors, self.random_vectors)
        results = []

        # Loop over the query hashes and determine if they exist in
        # the current table.
        #         for h in hashes:
        if hashes in self.table:
            results.extend(self.table[hashes])
        return results


class LSH:
    def __init__(self, hash_size, dim, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size, dim))

    def hashing(self, id, vectors, label):
        for table in self.tables:
            return table.hashing(id, vectors, label)

    def add(self, id, vectors, label, h):
        for table in self.tables:
            table.add(id, vectors, label, h)

    def query(self, vectors):
        results = []
        for table in self.tables:
            results.extend(table.query(vectors))
        return results


class BuildLSHTable:
    def __init__(
            self,
            dl_model,
            hash_size=8,
            dim=2048,
            num_tables=10,
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.dim, self.num_tables)
        self.model = dl_model

    def train(self, training_files, batch_size):
        loaded_imgs = []
        start_load_and_resize = datetime.now()
        for training_file in training_files:
            label, name = training_file
            img = image.load_img(name, target_size=(224, 224))
            loaded_imgs.append((label, img))
        end_load_and_resize = datetime.now()
        time_load_and_resize = end_load_and_resize - start_load_and_resize
        time_load_and_resize = time_load_and_resize.total_seconds()
        print(f'Load and resize  took {time_load_and_resize}')

        # process #time this
        start_process = datetime.now()
        imgs = []
        for loaded_img in loaded_imgs:
            lab, img = loaded_img
            xs = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
            imgs.append(xs)
        end_process = datetime.now()
        time_process = end_process - start_process
        time_process = time_process.total_seconds()
        print(f'processing  took {time_process}')
        # feature_extract #time this
        embeddings = []
        start_extract = datetime.now()
        em = self.model.predict(np.stack(imgs, axis=1)[0], batch_size=batch_size, verbose=1)
        embeddings.extend(em)
        end_extract = datetime.now()
        time_extract = end_extract - start_extract
        time_extract = time_extract.total_seconds()
        print(f'Feature Extraction  took {time_extract}')
        labels = [trainingfile[0] for trainingfile in training_files]
        hashes = []

        start_hash = datetime.now()
        for (id, feature) in enumerate(embeddings):
            embedding = feature
            h = self.lsh.hashing(id, embedding, labels[id])
            hashes.append((id, embedding, labels[id], h))
        end_hash = datetime.now()
        time_hash = end_hash - start_hash
        time_hash = time_hash.total_seconds()
        print(f'Hashing  took {time_hash}')
        start_add = datetime.now()
        for item in hashes:
            id, embedding, label, h = item
            self.lsh.add(id, embedding, label, h)
        end_add = datetime.now()
        time_add = end_add - start_add
        time_add = time_add.total_seconds()
        print(f'Adding to Table  took {time_add}')

    def query(self, path, verbose=True):
        # Compute the embeddings of the query image and fetch the results.
        img = image.load_img(path, target_size=(224, 224))
        xs = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        features = self.model.predict(xs)
        results = self.lsh.query(features[0])
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


def get_model():
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return resnet


resnet50 = get_model()

start = datetime.now()
lsh_builder = BuildLSHTable(resnet50)
lsh_builder.train(train, 1)
# lsh_builder.train(benchmark_data, 1)
end = datetime.now()
total = end - start
total = total.total_seconds()
print(f'Total time taken is{total}')

