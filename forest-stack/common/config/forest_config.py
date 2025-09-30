class Config:
    """Config class
    """
    PLANETARY_SUB_KEY = "XXXXXXXXX"
    ELEMENT84_STAC_URL = "https://earth-search.aws.element84.com/v1"
    PLANETARY_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    LULC_START_YEAR = 2023

    # Forest Canopy Density Classification Thresholds by State/Region
    FCD_THRESHOLDS = {
        "RJ": {
            "open_forest": {"min": 0, "max": 25},
            "low_density": {"min": 25, "max": 50},
            "medium_density": {"min": 50, "max": 75},
            "high_density": {"min": 75, "max": 100}
        }
    }

    @classmethod
    def get_fcd_thresholds(cls, state_code: str) -> dict:
        """Get FCD classification thresholds for a specific state
        
        Args:
            state_code (str): State code or 'default' for national thresholds
                            Options: default, RJ (Rajasthan)
                            
        Returns:
            dict: Dictionary containing threshold values for each forest density class
        """
        state_key = state_code.upper() if state_code != "default" else "default"
        return cls.FCD_THRESHOLDS.get(state_key, cls.FCD_THRESHOLDS["default"])
