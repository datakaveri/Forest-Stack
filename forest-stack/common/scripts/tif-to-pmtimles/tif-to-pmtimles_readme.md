# TIFF to PMTiles Conversion – Detailed Documentation

This module provides shell scripts for converting GeoTIFF files to PMTiles format, optimized for web mapping applications. It includes GDAL-based conversion with WebP compression and configurable tile settings for optimal performance and file size.

---

## 1. What the script does

### Conversion Pipeline

1. **TIFF to MBTiles**:
   - Converts GeoTIFF files to MBTiles format using GDAL
   - Applies WebP compression with lossless settings
   - Configures tile size and zoom levels for optimal performance

2. **MBTiles to PMTiles**:
   - Converts MBTiles to PMTiles format using pmtiles tool
   - Optimizes for web delivery and caching
   - Reduces file size while maintaining quality

---

## 2. Script Details

### Main Conversion Script
```bash
# Convert TIFF to MBTiles with WebP compression
GDAL_CACHEMAX=25600 rio mbtiles rj-humansettlements.tif rj-humansettlements.mbtiles \
    --format WEBP \
    --co LOSSLESS=TRUE \
    --co QUALITY=60 \
    --progress-bar \
    --zoom-levels 0..14 \
    --tile-size 512

# Convert MBTiles to PMTiles
./pmtiles convert rj-humansettlements.mbtiles rj-humansettlements.pmtiles
```

### Configuration Parameters

#### GDAL Settings
- **GDAL_CACHEMAX**: Memory cache size (25.6GB in example)
- **Format**: WEBP for optimal compression
- **Compression**: LOSSLESS=TRUE for quality preservation
- **Quality**: 60% for balanced size/quality

#### Tile Settings
- **Zoom Levels**: 0-14 (suitable for most web applications)
- **Tile Size**: 512x512 pixels (standard web tile size)
- **Progress Bar**: Shows conversion progress

---

## 3. Dependencies

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

## 4. Usage Instructions

### Basic Usage
```bash
# Make script executable
chmod +x tif-to-pmtimles.sh

# Run conversion
./tif-to-pmtimles.sh
```

### Custom Usage
```bash
# Convert specific file
GDAL_CACHEMAX=25600 rio mbtiles input.tif output.mbtiles \
    --format WEBP \
    --co LOSSLESS=TRUE \
    --co QUALITY=60 \
    --progress-bar \
    --zoom-levels 0..14 \
    --tile-size 512

./pmtiles convert output.mbtiles output.pmtiles
```

---

## 5. Configuration Options

### Memory Settings
- **GDAL_CACHEMAX**: Adjust based on available RAM
  - 25600 = 25.6GB
  - 12800 = 12.8GB
  - 6400 = 6.4GB

### Compression Settings
- **LOSSLESS=TRUE**: Preserves all data (larger files)
- **LOSSLESS=FALSE**: Lossy compression (smaller files)
- **QUALITY**: 0-100 (higher = better quality, larger files)

### Tile Settings
- **zoom-levels**: Adjust based on data resolution
  - 0..14: Standard web mapping
  - 0..18: High-resolution data
  - 0..10: Low-resolution data

- **tile-size**: Standard web tile sizes
  - 256: Standard size
  - 512: High-resolution displays
  - 1024: Very high-resolution displays

---

## 6. Output Formats

### MBTiles Format
- **File Extension**: `.mbtiles`
- **Format**: SQLite database with tile data
- **Compression**: WebP with configurable quality
- **Usage**: Compatible with most web mapping libraries

### PMTiles Format
- **File Extension**: `.pmtiles`
- **Format**: Optimized for web delivery
- **Compression**: Built-in compression
- **Usage**: Direct web serving without server setup

---

## 7. Performance Considerations

### Memory Usage
- **GDAL_CACHEMAX**: Higher values = faster processing, more memory
- **Tile Size**: Larger tiles = fewer files, more memory per tile
- **Zoom Levels**: More levels = more tiles, longer processing

### File Size Optimization
- **WebP Compression**: Significantly reduces file size
- **Quality Settings**: Balance between size and quality
- **Zoom Levels**: Limit to necessary levels only

### Processing Time
- **File Size**: Larger files take longer to process
- **Zoom Levels**: More levels = longer processing
- **Memory**: More memory = faster processing

---

## 8. Integration with Forest-Stack

### Web Mapping Applications
- **Forest Cover Maps**: Convert NDVI and forest mask data
- **Canopy Density**: Process canopy density rasters
- **Climate Data**: Convert rainfall and soil moisture data
- **Interactive Dashboards**: Enable web-based visualization

### Data Products
- **Web-ready Tiles**: Optimized for web delivery
- **Mobile-friendly**: Responsive tile loading
- **Caching**: Efficient browser caching
- **Performance**: Fast loading times

---

## 9. Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce GDAL_CACHEMAX value
2. **Disk Space**: Ensure sufficient space for output files
3. **Permission Errors**: Check file permissions and tool installation
4. **Format Errors**: Verify input TIFF file integrity

### Debug Tips
- Check available memory before processing
- Monitor disk space during conversion
- Verify input file format and projection
- Test with small files first

### Performance Issues
- **Slow Processing**: Increase memory allocation
- **Large Files**: Consider processing in chunks
- **Memory Problems**: Reduce cache size or tile size
- **Disk I/O**: Use SSD storage for better performance

---

## 10. Best Practices

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
- **Backup**: Keep original TIFF files as backup

---

## 11. Future Enhancements

### Potential Improvements
- **Batch Processing**: Process multiple files automatically
- **Quality Assessment**: Automated quality validation
- **Metadata**: Include source information in output
- **Compression**: Advanced compression algorithms

### Integration Opportunities
- **Automated Workflows**: Integrate with data processing pipelines
- **Cloud Processing**: Deploy on cloud platforms
- **API Integration**: Direct integration with web services
- **Monitoring**: Progress tracking and error reporting
