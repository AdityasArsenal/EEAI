import pandas as pd 
import re
from Config import *


def get_input_data()->pd.DataFrame:
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df = pd.concat([df1, df2])
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    df["y"] = df[Config.CLASS_COL]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df


def de_duplication(data: pd.DataFrame):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            ["L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            ["Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = "(On.{30,60}wrote:)"
    pattern_3 = "(Re\s?:|RE\s?:)"
    pattern_4 = "(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = "(\s?\*\*\*\*\*\(PHONE\))*$"

    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # -------- start processing ticket data

    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        #print(t)
        df = data.loc[data['Ticket id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df[Config.INTERACTION_CONTENT]:

            # print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            #print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated
    data.to_csv('out.csv')
    data[Config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data
