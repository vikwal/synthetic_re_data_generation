import os
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_features(data: pd.DataFrame, 
                 params: dict,
                 adj_params: dict
                ):
    # calculate pressure
    #pressure = pvlib.atmosphere.alt2pres(elevation)
    dhi = data[params['dhi']['param']]
    ghi = data[params['ghi']['param']]
    pressure = data[params['pressure']['param']]
    temperature = data[params['temperature']['param']]
    v_wind = data[params['v_wind']['param']]
    latitude = data[params['latitude']['param']]
    longitude = data[params['longitude']['param']]
    elevation = data[params['elevation']['param']]
    
    surface_tilt = adj_params['surface_tilt']
    surface_azimuth = adj_params['surface_azimuth']
    albedo = adj_params['albedo']
    
    # get solar position
    solpos = pvlib.solarposition.get_solarposition(
        time=data.index,
        latitude=latitude,
        longitude=longitude,
        altitude=elevation,
        temperature=temperature,
        pressure=pressure,
    )
    solar_zenith = solpos['zenith']
    solar_azimuth = solpos['azimuth']
    
    # GHI and DHI in W/m^2 --> J / cm^2 = J / 0,0001 m^2 = 10000 J / m^2 --> Dividing by 600 seconds (DWD is giving GHI as sum of 10 minutes))
    dhi = data[params['dhi']['param']] * 1e4 / 600
    ghi = data[params['ghi']['param']] * 1e4 / 600
    
    # get dni from ghi, dni and zenith
    dni = pvlib.irradiance.dni(ghi=ghi,
                               dhi=dhi,
                               zenith=solar_zenith)
    
    # get total irradiance
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(data.index),
        albedo=albedo,
        model='haydavies',
    )
    cell_temperature = pvlib.temperature.faiman(total_irradiance['poa_global'], 
                                                temperature,
                                                v_wind, 
                                                u0=25.0, 
                                                u1=6.84)
    return total_irradiance, cell_temperature


def generate_pv_power(total_irradiance: pd.Series,
                      cell_temperature: pd.Series,
                      adj_params: dict                      
                      ) -> pd.Series:
    
    installed_power = adj_params['installed_power']
    gamma_pdc = adj_params['gamma_pdc']

    #Check: The adj_params in config is eta_inv_nom not eta_env_noms
    eta_env_nom = adj_params['eta_inv_nom']
    eta_env_ref = adj_params['eta_inv_ref']
    
    power_dc = pvlib.pvsystem.pvwatts_dc(total_irradiance, 
                                         cell_temperature, 
                                         installed_power,
                                         gamma_pdc=gamma_pdc, 
                                         temp_ref=25.0)
    
    return pvlib.inverter.pvwatts(power_dc, 
                                  installed_power, 
                                  eta_inv_nom=eta_env_nom, 
                                  eta_inv_ref=eta_env_ref)
    
    
def plot_power_and_features(day: str, 
                            plot_names: list,
                            features: list,
                            power: pd.Series,
                            synchronize_axes=True,
                            save_fig=False,
                            streamlit=False
                            ): 

    day = pd.Timestamp(day)
    index_0 = power.index.get_loc(day)
    index_1 = power.index.get_loc(day + pd.Timedelta(days=1))
    date = str(features[0].index[index_0:index_1][0].date())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fontsize = 14
    lines = []
    title_suffix = ''
    
    # plot power
    line1, = ax1.plot(
    power[index_0:index_1],
    label="Power Output (W)",
    color="black",
    linewidth=2.0
    )
    lines.append(line1)

    # configure secondary y-axis
    ax1.set_xlabel("Time", fontsize=fontsize)
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize-2)
    
    ax2 = ax1.twinx()
    # plot irradiance components
    for name, series in zip(plot_names, features):
        line, = ax2.plot(
            series[index_0:index_1],
            label=f"{name} (W/m$^2$)",
            linestyle='--',
            linewidth=2.0
        )
        lines.append(line)

    # configure primary y-axis
    ax2.set_ylabel("Energy flux density (W/m$^2$)", fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)

    # Format x-axis to show only hours (HH)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[1:-1])


    # Synchronize y-axes
    if synchronize_axes:
        title_suffix = '(synched axes)'
        all_ghi_min = min([series[index_0:index_1].min() for series in features])
        all_ghi_max = max([series[index_0:index_1].max() for series in features])
        y_min = min(all_ghi_min, power[index_0:index_1].min())
        y_max = max(all_ghi_max, power[index_0:index_1].max())
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

    # legend
    lines.append(lines.pop(0))
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=fontsize)

    plt.title(f"Irradiance and Power Output on {date} {title_suffix}", fontsize=fontsize)
    fig.tight_layout()
    #plt.grid(True)
    if(streamlit):
        return fig
    if save_fig:
        save_path = 'figs/PV'
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f'{date}.png')
        plt.savefig(save_file, dpi=300)
        plt.close()        
    else:
        plt.show()


def plot_power_and_features_pv_streamlit(day: str, 
                            plot_names: list,
                            features: list,
                            power: pd.Series
                            ): 

    day = pd.Timestamp(day)
    index_0 = power.index.get_loc(day)
    index_1 = power.index.get_loc(day + pd.Timedelta(days=1))
    date = str(features[0].index[index_0:index_1][0].date())

    fig, ax1 = plt.subplots(figsize=(8, 4))

    font_properties = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 6}  
    color = '#999999'
    line_colors = ['#be95c4','#e0b1cb','#9f86c0']  
    fontsize = 8
    lines = []
    title_suffix = ''
    
    # plot power
    line1, = ax1.plot(
    power[index_0:index_1],
    label="Power Output (W)",
    color="#231942",
    linewidth=1.0
    )
    lines.append(line1)

    # configure secondary y-axis
    ax1.set_xlabel("Time", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.set_ylabel("Power Output (W)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize-2) 
    ax2 = ax1.twinx()

    color_index = 0

    # plot irradiance components
    for name, series in zip(plot_names, features):
        line, = ax2.plot(
            series[index_0:index_1],
            label=f"{name} (W/m$^2$)",
            linestyle='-',
            color=line_colors[color_index],
            linewidth=1.0
        )
        lines.append(line)
        color_index += 1

    # configure primary y-axis
    ax2.set_ylabel("Energy flux density (W/m$^2$)", fontsize=fontsize, color=color, family='DejaVu Sans')
    ax2.tick_params(axis='y', labelsize=fontsize)

    # Format x-axis to show only hours (HH)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ticks = ax1.get_xticks()
    ax1.set_xticks(ticks[1:-1])

    #Format chart to look similar to the streamlit charts
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    labels = [line.get_label() for line in lines]
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_color(color)

    # legend
    lines.append(lines.pop(0))
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc=10,
        bbox_to_anchor=(0.4, -0.2),
        ncol=2,
        frameon=False,  
        fontsize=8,
        labelcolor=color, 
        edgecolor='none'
    )
    ax1.grid(True, axis='y', color='gray', linewidth=0.5)
     
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)

    plt.title(f"Irradiance and Power Output on {date} {title_suffix}", fontsize=fontsize, color=color, family='DejaVu Sans', fontweight='bold')
    fig.tight_layout()

    return fig