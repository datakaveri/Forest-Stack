GDAL_CACHEMAX=25600 rio mbtiles rj-humansettlements.tif rj-humansettlements.mbtiles --format WEBP --co LOSSLESS=TRUE --co QUALITY=60 --progress-bar --zoom-levels 0..14 --tile-size 512

./pmtiles convert rj-humansettlements.mbtiles rj-humansettlements.pmtiles