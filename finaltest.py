import re
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import urllib
from astropy.visualization import make_lupton_rgb
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
import matplotlib.image as mpimg
from astropy.visualization import (imshow_norm,
                                   MinMaxInterval,
                                   AsinhStretch)
from astroquery.cadc import Cadc
import shutil
import tempfile
import os
import numpy
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import pylab
import matplotlib.patches as patches


# TODO/PROBLEMS
# add marker for pulsar on jpegs (SDSS, PS)
# ps/sdss images sometimes so populated image is indecipherable
# ra and dec on jpegs doesnt work  
# size of image constant, but no universal instance variable
# code sometimes just hangs or stops working for no apparent reason 
# if vlass image is split between two frames i.e crab nebula, will crash. need to change size in that case
# vlass image doesn't look very good  
# sometimes sdss returns wrong fits file/fits file without pulsar in range 
# 2mass array overlap error- probably an error in getting fits files from catalog.
# ^ this means the fits file doesnt cover the position of the pulsar

#TEXTFILE FORMAT
#ra(00h00m00s) dec(00d00m00s) name vlass 2mass sdss
#ex:18h53m57.318305s +13d03m44.05085s J1853+1303 vlass 2mass sdss

imgsize = u.Quantity((3.6,3.6), u.arcmin) #setting size for 2mass image 
file7 = open('pulsarlist4.txt')#ENTER TEXT FILE NAME 
data = file7.readlines()
RA = ''
DEC = ''
vlass = False 
twomass = False
sdss = False 
panstar = False
pulsarname = ""
SIZE = .03 #in degrees 
jband = ''
hband = ''
kband = ''
count = 0
dir = os.path.dirname(os.path.abspath(__file__))
start = 'https'
end = 'fits'
plt.rcParams.update({'font.size': 15})
radius = 0.03*u.deg #vlass image radius
savefile = '.'
boxsize = (15,15)

showgrid = False #if grids wanted on VLASS image- set to true 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def getimages(ra,dec,size=240,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table

def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    
    """Get color image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,dec,size=size,filters=filters,output_size=output_size,format=format,color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):#not used in current code, but can be easily adapted
    
    """Get grayscale image at a sky position
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """
    
    if format not in ("jpg","png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,dec,size=size,filters=filter,output_size=output_size,format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def open_VLASS_I_mJy(file):
    hdu = fits.open(file)
    extension = 0
    wcs = WCS(hdu[extension].header)
    image_data = hdu[extension].data[0,0,:,:]*1000
    # places image into mJy
    
    return wcs.celestial, image_data


def asinh_plot_VLASS_mJy(wcs_celestial, image_data, showgrid=False):
    cmap = plt.get_cmap('viridis')

    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(11, 8), dpi=75)
    ax = plt.subplot(projection=wcs_celestial)
    im, norm = imshow_norm(image_data, ax, origin='lower',
                       interval=MinMaxInterval(), 
                       stretch=AsinhStretch(),
                       vmin = -1e-5,
                       cmap=cmap)
    cbar = fig.colorbar(im, cmap=cmap)
    ax.set_xlabel('RA J2000')
    ax.set_ylabel('Dec J2000')
    ax.set_title(pulsarname)
    cbar.set_label('mJy')
    if showgrid:
        ax.grid(color='white', ls='solid')
    cutout = Cutout2D(image_data, position, boxsize,wcs=wcs_celestial)
    cutout.plot_on_original(color='white')
    plt.savefig(pulsarname)

    plt.show()


def plot_http_fits(url,showgrid=False,savefile=''):
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            shutil.copyfileobj(response, tmp_file)

            wcs, image_data = open_VLASS_I_mJy(tmp_file.name)
            asinh_plot_VLASS_mJy(wcs, image_data, showgrid)
            shutil.copy(tmp_file.name,pulsarname+'.fits')

            tmp_file.close()


def construct_cadc_url(baseurl, position, radius):
    ICRS_position = position.transform_to('icrs')
    basefile = baseurl.split('pub/')[1].split('?')[0]
    if (basefile[-10:] == 'subim.fits' and basefile[:6] == 'VLASS/'):
        url = ( 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/cutout?uri=ad:' 
               + urllib.parse.quote(basefile) 
               + ('&cutout=Circle+ICRS+{}+{}+{}').format(ICRS_position.ra.degree,
                                                         ICRS_position.dec.degree,
                                                         radius.to(u.degree).value))
        return url
    else:
        print('CADC URL appears to be incorrect: {}'.format(basefile))
        return None


def get_VLASS_images(position, radius, showgrid=False, savefile=''):

    cadc = Cadc()
    result = cadc.query_region(position, collection='VLASS')
    if len(result) == 0:
        print('No Data at CADC at position: {}'.format(position.to_string('hmsdms')))
    else:
        urls = cadc.get_data_urls(result)
        for url in urls:
            cutout_url = construct_cadc_url(url, position, radius)
            plot_http_fits(cutout_url, showgrid, savefile)


def process_cutout_input(position, radius, showgrid=False, savefile=''):
    radius_deg = radius.to(u.degree).value
    if position.dec.degree < -40:
        print('VLASS only covers Declinations > -40 deg')
    elif (radius_deg >1.5 or radius_deg < 0.0103):
        print('Radius must be between 0.0103 and 1.5 deg')
    else:
        get_VLASS_images(position, radius, showgrid, savefile)

    #code taken from astropy https://docs.astropy.org/en/stable/nddata/utils.html
def download_image_save_cutout(url, position, size,band):#cutouts for 2mass 
    filename = url
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    hdu.data = cutout.data
    hdu.header.update(cutout.wcs.to_header())
    if band == 'j':
        cutout_filename = pulsarname +' jband.fits'
        hdu.writeto(cutout_filename, overwrite=True)
    if band == 'h':
        cutout_filename = pulsarname +' hband.fits'
        hdu.writeto(cutout_filename, overwrite=True)
    if band == 'k':
        cutout_filename = pulsarname+ ' kband.fits'
        hdu.writeto(cutout_filename, overwrite=True)

def download_image_save_cutout2(url, position, size, band):#cutouts for sloan 
    # Download the image

    # Load the image and the WCS
    hdu = fits.open(url)[0]
    wcs = WCS(hdu.header)

    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    if band == 'r':
        cutout_filename = pulsarname +' r band.fits'
        hdu.writeto(cutout_filename, overwrite=True)
    if band == 'g':
        cutout_filename = pulsarname +' g band.fits'
        hdu.writeto(cutout_filename, overwrite=True)
    if band == 'u':
        cutout_filename = pulsarname+ ' u band.fits'
        hdu.writeto(cutout_filename, overwrite=True)

for line in data:#main loop where everything runs 
    RA = line.split()[0]
    DEC = line.split()[1]
    position = SkyCoord(RA + ' ' + DEC, frame='icrs')
    pulsarname = line.split()[2]
    if line.split()[3] == 'vlass':#these if statements check which surveys to get images from 
        vlass = True 
    if line.split()[4] == '2mass':
        twomass = True 
    if line.split()[5] == 'sdss':
        sdss = True 
    panstar = False
    os.chdir(dir)
    createFolder(pulsarname)
    xmlurl = 'https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?POS=+'+ RA+','+ DEC + '&SIZE=.03' #2mass url 
    file = urllib.request.urlopen(xmlurl)
    xml = file.readlines()
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/' + pulsarname)
    xmlurlsloan = 'http://skyserver.sdss.org/dr14/SkyServerWS/SIAP/getSIAP?POS=+'+str(position.ra.deg)+','+str(position.dec.deg)+'&SIZE=.03&FORMAT=image/fits&bandpass=*' #sdss url generation
    file2 = urllib.request.urlopen(xmlurlsloan) 
    xml1 = file2.readlines()
    xml1.reverse()

    if vlass:
        process_cutout_input(position, radius, showgrid, savefile)
    
    
    if twomass: #2mass code 
        count = 0
        for line in xml:
            if b'https' and b'.fits' in line: #finding fits files 
                count+= 1
                if count == 1: #jband
                    string2 = line.decode("utf-8")
                    jband = string2[string2.find(start)+len(start):string2.rfind(end)] 
                if count == 2: #hband 
                    string2 = re.search('https(.+?)fits', line.decode("utf-8"))
                    hband = string2.group(1)
                if count == 3: #kband
                    string3 = re.search('https(.+?)fits', line.decode("utf-8"))
                    kband = string3.group(1)
                    

        jbandurl = 'https' + jband + 'fits'#finishing url construction 
        hbandurl = 'https' + hband + 'fits'
        kbandurl = 'https' + kband + 'fits'

        plt.figure(1,figsize=(11,8), dpi = 75)
        plt.rcParams.update({'font.size': 15})
        download_image_save_cutout(jbandurl,position,imgsize,'j')#saves fits files for plotting
        download_image_save_cutout(hbandurl,position,imgsize,'h')
        download_image_save_cutout(kbandurl,position,imgsize,'k')

        #IMAGE CUTOUT PLOTTING
        wcs = WCS(fits.open(pulsarname +' jband.fits')[0].header)
        axes = plt.subplot(projection=wcs)
        plt.title(pulsarname + ' J BAND')
        axes.set_xlabel('RA J2000')
        axes.set_ylabel('Dec J2000')
        plt.imshow(fits.open(pulsarname +' jband.fits')[0].data, origin = 'lower', cmap = 'gist_heat', norm= LogNorm())
        cutout = Cutout2D(fits.open(pulsarname +' jband.fits')[0].data, position, boxsize,wcs=wcs)
        cutout.plot_on_original(color='white')
        plt.savefig(pulsarname+ ' jband.png', overwrite=True)
        plt.show(1)

        plt.figure(2,figsize=(11,8), dpi = 75)
        axes = plt.subplot(projection=wcs)
        plt.title(pulsarname + ' H BAND')
        axes.set_xlabel('RA J2000')
        axes.set_ylabel('Dec J2000')
        plt.imshow(fits.open(pulsarname +' hband.fits')[0].data, origin = 'lower', cmap = 'gist_heat', norm= LogNorm())
        cutout = Cutout2D(fits.open(pulsarname +' hband.fits')[0].data, position, boxsize,wcs=wcs)
        cutout.plot_on_original(color='white')
        plt.savefig(pulsarname+ ' hband.png', overwrite=True)

        plt.show(2)

        plt.figure(3,figsize=(11,8), dpi = 75)
        axes = plt.subplot(projection=wcs)
        plt.title(pulsarname + ' K BAND')
        axes.set_xlabel('RA J2000')
        axes.set_ylabel('Dec J2000')
        plt.imshow(fits.open(pulsarname +' kband.fits')[0].data, origin = 'lower', cmap = 'gist_heat', norm= LogNorm())
        cutout = Cutout2D(fits.open(pulsarname +' kband.fits')[0].data, position, boxsize,wcs=wcs)
        cutout.plot_on_original(color='white')
        plt.savefig(pulsarname+ ' kband.png')

        plt.show(3)

    if sdss: #SDSS code 
        count = 0 
        uband = ''
        gband = ''
        rband = ''

        for line in xml1: 
            if b'fits' in line: #dont know why this doesnt work without nested if statements... 
                if count < 3:
                    if b'http' in line:
                        if b'frame-r' in line:
                            rband = line.decode("utf-8")
                            count += 1
                        if b'frame-g' in line:
                            gband = line.decode('utf-8')
                            count += 1
                        if b'frame-u' in line:
                            uband = line.decode('utf-8')
                            count += 1

        if rband == '':
            print ('Position not in SDSS catalog. Using Pan-STARRS1 instead')#error checking. if image not in sloan goes to PS
            panstar = True
        else:
            uband = re.search('>(.+?)<', uband).group(1) #extracting url from line
            gband = re.search('>(.+?)<', gband).group(1)
            rband = re.search('>(.+?)<', rband).group(1)

            download_image_save_cutout2(uband,position,imgsize, 'u')#saving fits files 
            download_image_save_cutout2(gband,position,imgsize, 'g')
            download_image_save_cutout2(rband,position,imgsize, 'r')

            position = SkyCoord(RA + ' ' + DEC, frame='icrs')
            xmlurlsloan = 'http://skyserver.sdss.org/dr14/SkyServerWS/SIAP/getSIAP?POS=+'+str(position.ra.deg)+','+str(position.dec.deg)+'&SIZE=.03&FORMAT=image/jpeg&bandpass=*' #sdss url generation
            file = urllib.request.urlopen(xmlurlsloan) 
            xml = file.readlines()
            count = 0 
            sloanjpeg = ''
            for line in xml:
                if b'scale' in line: #dont know why this doesnt work without nested if statements... 
                    if b'http' in line:
                        count+= 1
                        if count == 1:
                            sloanjpeg = line.decode('utf-8')#finding line with url 

                           
            sloanjpegurl = re.search('>(.+?)<', sloanjpeg).group(1) #extracting url from line
            sloanjpeg = sloanjpegurl.replace("amp;", "")#formatting 
            urllib.request.urlretrieve(sloanjpeg,pulsarname+' SDSS no plot.jpg')#saving image 

            #wcs = WCS(fits.open(pulsarname +' u band.fits')[0].header)#plotting image
            plt.figure(4,figsize=(11,8), dpi = 75)
            #axes = plt.subplot(projection=wcs)
            plt.title(pulsarname)
            img=mpimg.imread(pulsarname+' SDSS no plot.jpg')
            plt.imshow(img)
            plt.savefig(pulsarname+ ' SDSS.png')
            plt.show(4)

    if panstar:#Pan-STARRS1 code
    	cim = getcolorim(position.ra.deg,position.dec.deg,size=640,filters="grz",format = 'jpg')
    	plt.figure(4,figsize=(11,8),dpi=75)
    	plt.title(pulsarname)
    	plt.imshow(cim)
    	plt.savefig(pulsarname+ ' Pan-STARRS1.png')
    	plt.show(5)
    	ps1fits = geturl(position.ra.deg,position.dec.deg,size=640,filters="grz",format = 'fits')
    	urllib.request.urlretrieve(ps1fits[0], pulsarname + ' PS1 g band.fits')
    	urllib.request.urlretrieve(ps1fits[1], pulsarname + ' PS1 r band.fits')
    	urllib.request.urlretrieve(ps1fits[2], pulsarname + ' PS1 z band.fits')

	    

