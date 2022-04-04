import json

schema = json.dumps({"namespace": "file.avro",
                     "type": "record",
                     "name": "label",
                     "fields": [
                         {"name": "label", "type": "string"},
                         {"name": "id", "type": "string"},
                         {"name": "image_array", "type": ["bytes", "string"]}
                     ]
                     })
hash_schema = json.dumps({"namespace": "hash.avro",
                          "type": "record",
                          "name": "label",
                          "fields": [
                              {"name": "label", "type": "string"},
                              {"name": "id", "type": "string"},
                              {"name": "hash_value", "type": ["int", "string"]}
                          ]
                          })
