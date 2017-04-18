import struct
import zlib

def encode(buf, width, height):
  """ buf: must be bytes or a bytearray in py3, a regular string in py2. formatted RGBRGB... """
  assert (width * height * 3 == len(buf))
  bpp = 3

  def raw_data():
    # reverse the vertical line order and add null bytes at the start
    row_bytes = width * bpp
    for row_start in range((height - 1) * width * bpp, -1, -row_bytes):
      yield b'\x00'
      yield buf[row_start:row_start + row_bytes]

  def chunk(tag, data):
    return [
        struct.pack("!I", len(data)),
        tag,
        data,
        struct.pack("!I", 0xFFFFFFFF & zlib.crc32(data, zlib.crc32(tag)))
      ]

  SIGNATURE = b'\x89PNG\r\n\x1a\n'
  COLOR_TYPE_RGB = 2
  COLOR_TYPE_RGBA = 6
  bit_depth = 8
  return b''.join(
      [ SIGNATURE ] +
      chunk(b'IHDR', struct.pack("!2I5B", width, height, bit_depth, COLOR_TYPE_RGB, 0, 0, 0)) +
      chunk(b'IDAT', zlib.compress(b''.join(raw_data()), 9)) +
      chunk(b'IEND', b'')
    )
