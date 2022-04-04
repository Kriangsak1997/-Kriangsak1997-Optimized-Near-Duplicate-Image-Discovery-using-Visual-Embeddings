import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
import io


def dReader(msg, schm=None):
    _bw = io.BytesIO(msg)
    _bw.mode = 'b'
    if not schm:
        _br = DataFileReader(_bw, DatumReader())
    else:
        _br = DataFileReader(_bw, DatumReader(readers_schema=avro.schema.parse(schm)))
    _sch = _br.meta.get('avro.schema').decode('utf-8')
    _val = []
    for _x in _br:
        _val.append(_x)
    return (_sch, _val)


def dWriter(schm, msg):
    _bw = io.BytesIO()
    _dw = DataFileWriter(_bw, DatumWriter(), avro.schema.parse(schm))
    for _x in msg:
        _dw.append(_x)
    _dw.flush()
    _bw.seek(0)
    _val = _bw.read()
    _bw.close()
    return _val
