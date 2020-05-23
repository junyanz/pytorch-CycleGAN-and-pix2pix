import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


disaster_type_names = {
    0: 'flooding-0',
    1: 'flooding-1',
    2: 'flooding-2',
    3: 'flooding-3',
    4: 'fire-0',
    5: 'fire-1',
    6: 'fire-2',
    7: 'fire-3',
    8: 'wind-0',
    9: 'wind-1',
    10: 'wind-2',
    11: 'wind-3',
    12: 'tsunami-0',
    13: 'tsunami-1',
    14: 'tsunami-2',
    15: 'tsunami-3',
    16: 'earthquake-0',
    17: 'earthquake-1',
    18: 'earthquake-2',
    19: 'earthquake-3',
    20: 'volcano-0',
    21: 'volcano-1',
    22: 'volcano-2',
    23: 'volcano-3'
}

class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, labels=[], width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                if len(labels) > 0:
                    for im, txt, link, label in zip(ims, txts, links, labels):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt + str(label[0]) + str(disaster_type_names[int(label[0])]))
                else:
                    for im, txt, link in zip(ims, txts, links):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
