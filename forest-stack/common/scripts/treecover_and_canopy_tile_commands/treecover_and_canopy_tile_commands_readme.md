# Tree Cover and Canopy Density Tile Commands – Detailed Documentation

This module provides comprehensive shell scripts for processing tree cover and canopy density data into web-ready tile formats. It handles color-relief application, MBTiles generation, and PMTiles conversion for both tree cover and canopy density datasets across multiple years.

---

## 1. What the script does

### Processing Pipeline

1. **Color Relief Application**:
   - Applies color tables to raw raster data
   - Converts single-band data to RGB format
   - Optimizes for web visualization

2. **MBTiles Generation**:
   - Converts RGB TIFF files to MBTiles format
   - Applies WebP compression for optimal file size
   - Configures zoom levels and tile sizes

3. **PMTiles Conversion**:
   - Converts MBTiles to PMTiles format
   - Optimizes for web delivery and caching
   - Enables direct web serving

---

## 2. Data Processing

### Tree Cover Processing (2018-2024)
```bash
# Process each year from 2018 to 2024
for year in 2018 2019 2020 2021 2022 2023 2024; do
    # Apply color relief
    GDAL_CACHEMAX=2560 gdaldem color-relief \
        -co COMPRESS=LZW -alpha -of GTiff \
        ./data/treecover_tifs/${year}.tif \
        treecover-colortable.txt \
        ./data/treecover_tifs/${year}_rgb.tif
    
    # Convert to MBTiles
    GDAL_CACHEMAX=2560 rio mbtiles \
        ./data/treecover_tifs/${year}_rgb.tif \
        ./data/treecover_tifs/yforest_cover_${year}.mbtiles \
        --format WEBP \
        --co LOSSLESS=TRUE \
        --co QUALITY=60 \
        --progress-bar \
        --zoom-levels 0..14 \
        --tile-size 512
    
    # Convert to PMTiles
    ./pmtiles convert \
        ./data/treecover_tifs/yforest_cover_${year}.mbtiles \
        ./data/treecover_tifs/forest_cover_${year}.pmtiles
done
```

### Canopy Density Processing (2013-2024)
```bash
# Process each year from 2013 to 2024
for year in 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
    # Apply color relief
    GDAL_CACHEMAX=2560 gdaldem color-relief \
        -co COMPRESS=LZW -alpha -of GTiff \
        ./data/canopydensity/${year}_fcd_rcd.img \
        canopydensity-colormap.txt \
        ./data/canopydensity/${year}_rgb.tif
    
    # Convert to MBTiles
    GDAL_CACHEMAX=2560 rio mbtiles \
        ./data/canopydensity/${year}_rgb.tif \
        ./data/canopydensity/canopy_density_${year}.mbtiles \
        --format WEBP \
        --co LOSSLESS=TRUE \
        --co QUALITY=60 \
        --progress-bar \
        --zoom-levels 0..14 \
        --tile-size 512
    
    # Convert to PMTiles
    ./pmtiles convert \
        ./data/canopydensity/canopy_density_${year}.mbtiles \
        ./data/canopydensity/canopy_density_${year}.pmtiles
done
```

---

## 3. Configuration Parameters

### GDAL Settings
- **GDAL_CACHEMAX**: Memory cache size (2.56GB)
- **COMPRESS**: LZW compression for intermediate files
- **ALPHA**: Adds alpha channel for transparency
- **OF**: Output format (GTiff)

### MBTiles Settings
- **Format**: WEBP for optimal compression
- **Compression**: LOSSLESS=TRUE for quality preservation
- **Quality**: 60% for balanced size/quality
- **Zoom Levels**: 0-14 (standard web mapping)
- **Tile Size**: 512x512 pixels

### Color Tables
- **Tree Cover**: `treecover-colortable.txt`
- **Canopy Density**: `canopydensity-colormap.txt`

---

## 4. Data Sources

### Tree Cover Data
- **Source**: Forest cover datasets
- **Format**: Single-band TIFF files
- **Years**: 2018-2024
- **Naming**: `{year}.tif`

### Canopy Density Data
- **Source**: Forest Canopy Density (FCD) datasets
- **Format**: IMG files with FCD data
- **Years**: 2013-2024
- **Naming**: `{year}_fcd_rcd.img`

---

## 5. Output Structure

### Tree Cover Outputs
```
data/treecover_tifs/
├── 2018_rgb.tif
├── yforest_cover_2018.mbtiles
├── forest_cover_2018.pmtiles
├── 2019_rgb.tif
├── yforest_cover_2019.mbtiles
├── forest_cover_2019.pmtiles
└── ...
```

### Canopy Density Outputs
```
data/canopydensity/
├── 2013_rgb.tif
├── canopy_density_2013.mbtiles
├── canopy_density_2013.pmtiles
├── 2014_rgb.tif
├── canopy_density_2014.mbtiles
├── canopy_density_2014.pmtiles
└── ...
```

---

## 6. Dependencies

### Required Tools
- **GDAL**: Geospatial Data Abstraction Library
- **rio**: Rasterio command-line interface
- **pmtiles**: PMTiles conversion tool

### Installation
```bash
# Install GDAL
conda install -c conda-forge gdal

# Install rasterio (includes rio)
pip install rasterio

# Install pmtiles
# Download from: https://github.com/protomaps/PMTiles
```

---

## 7. Usage Instructions

### Basic Usage
```bash
# Make script executable
chmod +x treecover_and_canopy_tile_commands.sh

# Run processing
./treecover_and_canopy_tile_commands.sh
```

### Custom Usage
```bash
# Process specific year
year=2023
GDAL_CACHEMAX=2560 gdaldem color-relief \
    -co COMPRESS=LZW -alpha -of GTiff \
    ./data/treecover_tifs/${year}.tif \
    treecover-colortable.txt \
    ./data/treecover_tifs/${year}_rgb.tif

GDAL_CACHEMAX=2560 rio mbtiles \
    ./data/treecover_tifs/${year}_rgb.tif \
    ./data/treecover_tifs/yforest_cover_${year}.mbtiles \
    --format WEBP \
    --co LOSSLESS=TRUE \
    --co QUALITY=60 \
    --progress-bar \
    --zoom-levels 0..14 \
    --tile-size 512

./pmtiles convert \
    ./data/treecover_tifs/yforest_cover_${year}.mbtiles \
    ./data/treecover_tifs/forest_cover_${year}.pmtiles
```

---

## 8. Performance Considerations

### Memory Usage
- **GDAL_CACHEMAX**: 2.56GB per process
- **Parallel Processing**: Multiple years can be processed simultaneously
- **Disk Space**: Ensure sufficient space for all output files

### Processing Time
- **File Size**: Larger files take longer to process
- **Years**: More years = longer total processing time
- **Memory**: More memory = faster processing

### File Size Optimization
- **WebP Compression**: Significantly reduces file size
- **Quality Settings**: Balance between size and quality
- **Zoom Levels**: Limit to necessary levels only

---

## 9. Integration with Forest-Stack

### Web Mapping Applications
- **Forest Monitoring**: Interactive forest cover maps
- **Canopy Analysis**: Canopy density visualization
- **Temporal Analysis**: Year-over-year comparisons
- **Dashboard Integration**: Web-based forest management tools

### Data Products
- **Web-ready Tiles**: Optimized for web delivery
- **Mobile-friendly**: Responsive tile loading
- **Caching**: Efficient browser caching
- **Performance**: Fast loading times

---

## 10. Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce GDAL_CACHEMAX value
2. **Disk Space**: Ensure sufficient space for output files
3. **Permission Errors**: Check file permissions and tool installation
4. **Format Errors**: Verify input file integrity

### Debug Tips
- Check available memory before processing
- Monitor disk space during conversion
- Verify input file format and projection
- Test with single year first

### Performance Issues
- **Slow Processing**: Increase memory allocation
- **Large Files**: Consider processing in chunks
- **Memory Problems**: Reduce cache size or tile size
- **Disk I/O**: Use SSD storage for better performance

---

## 11. Best Practices

### File Preparation
- **Projection**: Ensure consistent projection (Web Mercator recommended)
- **Resolution**: Optimize resolution for target zoom levels
- **Data Type**: Use appropriate data types for size optimization

### Conversion Settings
- **Quality**: Start with 60% quality, adjust based on needs
- **Zoom Levels**: Limit to necessary levels only
- **Tile Size**: Use 512x512 for high-resolution displays

### Output Management
- **File Naming**: Use descriptive names for easy identification
- **Storage**: Organize output files in logical directory structure
- **Backup**: Keep original files as backup

---

## 12. Future Enhancements

### Potential Improvements
- **Batch Processing**: Process multiple years automatically
- **Quality Assessment**: Automated quality validation
- **Metadata**: Include source information in output
- **Compression**: Advanced compression algorithms

### Integration Opportunities
- **Automated Workflows**: Integrate with data processing pipelines
- **Cloud Processing**: Deploy on cloud platforms
- **API Integration**: Direct integration with web services
- **Monitoring**: Progress tracking and error reporting
