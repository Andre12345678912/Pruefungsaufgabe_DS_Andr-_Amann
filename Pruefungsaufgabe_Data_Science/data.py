import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame



def read_raw_data(low_file: Path, middle_file: Path, high_file: Path) -> dict:
    """
    - Lese die Rohdaten, welche in verschiedene Dateien für niedrige, mittlere und hohe Bildung als Datenrahmen.

    - Args:
    
        low_file: Niedrige Bildung.
        
        middle_file: Mittlere Bildung.
        
        high_file: Hohe bildung.

  -   Returns:
     
     -    dict: Dict beinahltet Datenrahmen für die Arbeitslosenrate von niedriger, mittlerer und hoher Bildung.
        
    """
    
    
    
    
    
    ret = {
        "Low": pd.read_csv(low_file),
        
        "Middle": pd.read_csv(middle_file),
        
        "High": pd.read_csv(high_file)
    } 
    
    return ret






def merge_and_clean_data(eurostrat_data: dict, oecd_data: dict) -> DataFrame:
    """
   
    - Extrahiere die benötigten Daten und konvertiere die Datentypen
    
    
   
   -  Merge die Daten in einen einzigen normalisierten Datenrahmen

    - Args:
        eurostrat_data: Daten von eurostat.
        
        oecd_data: Daten von OECD.

   -  Returns:
        DataFrame: Datenrahmen der gemergten Daten.
    """
    
    
    
    
    
    ret = pd.DataFrame(columns=["Year",  "USA - Low", "USA - Middle", "USA - High", "Germany - Low", "Germany - Middle", "Germany - High"])
    ret["USA - Low"] = ret["USA - Low"].astype(float) 
    ret["USA - Middle"] = ret["USA - Middle"].astype(float)
    ret["USA - High"] = ret["USA - High"].astype(float)
    ret["Germany - Low"] = ret["Germany - Low"].astype(float)
    ret["Germany - Middle"] = ret["Germany - Middle"].astype(float)
    ret["Germany - High"] = ret["Germany - High"].astype(float)
    ret["Year"] = oecd_data["Low"]["TIME"][oecd_data["Low"]["LOCATION"] == "USA"].astype(int).values

    # Extrahiere USA
    ret["USA - Low"]  = oecd_data["Low"]["Value"][oecd_data["Low"]["LOCATION"] == "USA"].astype(float).values
    ret["USA - Middle"] = oecd_data["Middle"]["Value"][oecd_data["Middle"]["LOCATION"] == "USA"].astype(float).values
    ret["USA - High"] = oecd_data["High"]["Value"][oecd_data["High"]["LOCATION"] == "USA"].astype(float).values

    # Extrahiere Deutschland und Merge
    ret["Germany - Low"] = (eurostrat_data["Low"]["Value"][eurostrat_data["Low"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values + \oecd_data["Low"]["Value"][oecd_data["Low"]["LOCATION"] == "DEU"].astype(float).values) / 2
    ret["Germany - Middle"] = (eurostrat_data["Middle"]["Value"][eurostrat_data["Middle"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values +  \oecd_data["Middle"]["Value"][oecd_data["Middle"]["LOCATION"] == "DEU"].astype(float).values) / 2
    ret["Germany - High"] = (eurostrat_data["High"]["Value"][eurostrat_data["High"]["GEO"] == "Germany (until 1990 former territory of the FRG)"].astype(float).values + \oecd_data["High"]["Value"][oecd_data["High"]["LOCATION"] == "DEU"].astype(float).values) / 2
    ret["Year"] = pd.to_datetime(ret["Year"], format="%Y").dt.to_period("Y")
    return ret







def plot_data(df: DataFrame, country: str, show=True):
    """
    
    - Plot Kurven für niedrige, mittlere und hohe Bildung eines Landes.

    - Args:
        df: Datenrahmen der alle Informationen beinhaltet
       
        country: Plot Daten dür ein spezifisches Land.
        
        show: Rufe plt.show() auf oder auch nicht.
    """
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent")
    
    df_c = df.copy()
    df_c["Year"] = df_c["Year"].dt.year
    
    plt.title(F"Unemployment rates for {country}")
    plt.plot("Year", F"{country} - Low", data=df_c, label=F"{country} | Low education")
    plt.plot("Year", F"{country} - Middle", data=df_c, label=F"{country} | Middle education")
    plt.plot("Year", F"{country} - High", data=df_c, label=F"{country} | High education")
    
    if show: 
        plt.legend()
        plt.show()
        
        
        
        


def plot_single_data(df: DataFrame, key: str, show=True):
    """
   
    - Plot Single Kurve für Daten die per Schlüssel definiert werden.

    - Args:
        df: Datenrahmen der alle Informationen beinhaltet
        
        key: Schlüssel des Datenrahmens zum plot
        
        show: Rufe plt.show() auf oder auch nicht.
        
    """
    
    
    
    
    
    
    plt.title(F"Unemployment rate for {key}")
    plt.xlabel("Year")
    plt.ylabel(F"Unemployed population in percent")
    plt.title(key)
    
    
    df_c = df.copy()
    df_c["Year"] = df_c["Year"].dt.year
    
    
    plt.plot("Year", key, data=df_c)
    if show:
        plt.show()
        
        
        
        
        


def pearson_r(x, y):
    """
    
    - Verrechne die Pearson Korrelations Koeffizienten zwischen den zwei Arrays.
    
    - Args:
      
        x: Daten für die X-Achse
      
        y: Daten für die Y Achse

    - Returns:
        float: Pearson r für x und y.
    """
    
    
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0,1]







def plot_linear_regression(x, y, title, show=True):
    """
    
    - Plot Datensatz mit linearer Regressionslinie

    - Args:
        
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
       
        title: Titel des Plot
        
        show: show: Rufe plt.show() auf oder auch nicht.
    """
    
    
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent") 
    plt.plot(x, y)
    plt.title(title)

    
    
    
    
     # eindimensional
    a, b = np.polyfit(x, y, 1)
    model_1d = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, model_1d(x), linewidth=2, color="red", label="Ployfit | 1D")
     # dreidimensional
    model_3d = np.poly1d(np.polyfit(x, y, 3)) 
    plt.plot(x, model_3d(x), linewidth=2, color="yellow", label="Ployfit | 3D")
    if show:
        print(title, "\nSlope: ", a, " Intercept: ", b)
        plt.legend()
        plt.show()

        
        
        
        
        

def calculate_slope(x, y):
    """
    - Berechne die Neigung zwischen X und Y.

    - Args:
    
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
        
    - Returns:
        float: Neigung der Datensätze.
    """
    
    
    slope = ((x-x.mean())*(y-y.mean())).sum()/(np.square(x-x.mean())).sum()
    return(slope)







def calculate_interception_point(x, y, m):
    """
   
   - Berechne den Abfangpunkt von X und Y mit slope m

   - Args:
   
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
        
        m: Slope.

    Returns:
    
        float: Abfangpunkt mit der Y Achse.
    """
    
    
    c = y.mean()-(m*x.mean())
    return(c)







def calculate_rss(x, y):
    """
    - Berechne RSS für x und y.

    - Args:
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse

   - Returns:
        float: RSS von x und y.
    """
    
    
    slope = calculate_slope(x, y)
    intercept = calculate_interception_point(x, y, slope)
    return rss(x, y, slope, intercept)







def rss(x, y, slope, intercept):
    """
    
    - Berechne RSS für x und y mit slope und Abfangpunkt

    - Args:
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
        
        slope: Slope.
        
        intercept: Abfangpunkt mit y-Achse.
    
    - Returns: 
    
        float: RSS.
    """
    
    
    return np.sum(np.square(y - slope * x - intercept))







def slope_vs_rss(x, y):
    """
    - Optimiersierungsfunktion für slope vs RSS

    - Args:
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
    """
    
    
    
    a_vals = np.linspace(-1, 1, 100) # slopes
    rss_array = np.empty_like(a_vals) # Arry initialisierung RSS
    slope = calculate_slope(x, y)
    intercept = calculate_interception_point(x, y, slope)
    for i, a in enumerate(a_vals):
        rss_array[i] = rss(x, y, a, intercept) # Berechne alle RSS Werte über slopes
    best_rss = rss_array.min() # minimalfunktion
    best_slope = float(a_vals[np.where(rss_array == best_rss)])
    plt.title("Slope vs RSS")
    plt.xlabel("Slope")
    plt.ylabel("RSS")
    plt.plot(a_vals, rss_array)
    plt.plot(best_slope, best_rss, "ro", label=F"Minimum [{best_slope} | {int(best_rss)}]")
    plt.legend()
    plt.show()

    
    
    
    
    

def linear_regression_vs_rss(x, y, title, show=True):
    """
  
    - Lineare Regression vs RSS

    Args:
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
        
        title: Titel des Plot.
        
        show: show: Rufe plt.show() auf oder auch nicht.
    """
    
    
    plt.xlabel("Year")
    plt.ylabel("Unemployed population in percent") 
    plt.plot(x, y)
    plt.title(title)
    slope, intercept = np.polyfit(x, y, 1)
    model_1d = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, model_1d(x), linewidth=2, color="red", label="Ployfit | 1D")
    rss_value = calculate_rss(x, y)
    points = x.max() - x.min() + 1 # zeichne Reste
    a_vals = np.linspace(x.min(), x.max(), points)
    residuals = np.empty_like(a_vals)
    for i, a in enumerate(a_vals): # berechne Reste
        residuals[i] = y[i] - (slope * a + intercept)
    label = None
    for i, residual in enumerate(residuals):
        if i == 0: label = "Residuals"
        else: label = None
        plt.plot([a_vals[i], a_vals[i]], [y[i], y[i] - residual], linewidth=0.5, alpha=0.5, color="red", label=label)
    if show:
        print(title, "\nRSS: ", rss_value)
        plt.legend()
        plt.show() 
        
        
        
        
        


def pairs_bootstrap(x, y, size=1):
    """
   - Führe Bootstrap über X und Y Daten aus

   - Args:
        x: Daten für die x-Achse
        
        y: Daten für die y-Achse
        
        size: Anzahl Bootstrap Durchläufe.
    
    - Returns:
        tuple: Alle berechneten slopes und Abfangpunkte.
    """
    
    
    inds = np.arange(len(x))
    slopes = []
    intercepts = []
    for _ in range(size):
        index = np.random.choice(inds, len(inds)) # zufällige Daten holen
        xx, yy = x[index], y[index]
        slope, intercept = np.polyfit(xx, yy, 1) # slope u intercept berechnen
        slopes.append(slope)
        intercepts.append(intercept)
        # append
    
    
    return slopes, intercepts







def pearson_coefficient_hypothesis_test(permute_data, fix_data, func=pearson_r, size=1000):
    """
    
    - Test, ob permute_data und fix_data unabhängig von einander sind
    - Hypothesentest (p werte) in Kombination mit pearson r.

    - Args:
        
        permute_data : Daten die permutiert sind.
        
        fix_data: Daten die nicht permutiert sind.
       
        func: Funktion zum berechnen spezifischer doppelter Werte
        
        size: Anzahl der Durchläufe.
    """
    
    
  
    pearson_observed = func(permute_data, fix_data) # Pearson r über data set
    print("Observed pearson: ", pearson_observed)
    replicates = np.empty(size)
    for i in range(size): # Berechne doppelte Werte
        permutated = np.random.permutation(permute_data)
        replicates[i] = func(permutated, fix_data)
    p_value = np.sum(replicates >= pearson_observed) / size # Berechne p Werte
    print("P-Value: ", p_value)
