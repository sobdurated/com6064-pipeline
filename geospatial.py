"""
geospatial.py
-------------
Pipeline step : geospatial
Contract      : run(input_data, context) -> GeoJSON FeatureCollection dict
Author        : Adel Ugur (Final Version - Multi-Model + Schema-Corrected Lookup)

Responsibilities:
  1. Extract and tag province + district for any untagged posts.
  2. Aggregate sentiment at BOTH province level and district level.
  3. Support multi-model nested schema (sentiment.llm & sentiment.transformer).
  4. HOTFIX: Join with `posts_raw` via the internal `_id` to retrieve dropped category tags instantly.
  5. Compute map-coloring scores including category-specific breakdowns for each model.
  6. Return a GeoJSON-compatible FeatureCollection.
"""

from __future__ import annotations

import re
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pymongo import UpdateOne

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TURKEY PROVINCE + DISTRICT REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

TURKEY_LOCATIONS = {
    "adana":           {"official": "Adana",           "districts": ["seyhan", "çukurova", "yüreğir", "sarıçam", "kozan", "ceyhan", "karataş", "feke", "imamoglu", "karaisalı", "pozantı", "saimbeyli", "tufanbeyli"]},
    "adıyaman":        {"official": "Adıyaman",        "districts": ["merkez", "besni", "çelikhan", "gerger", "gölbaşı", "kahta", "samsat", "sincik", "tut"]},
    "afyonkarahisar":  {"official": "Afyonkarahisar",  "aliases": ["afyon"], "districts": ["merkez", "bolvadin", "çay", "dinar", "emirdağ", "sandıklı", "sultandağı"]},
    "ağrı":            {"official": "Ağrı",            "districts": ["merkez", "diyadin", "doğubayazıt", "eleşkirt", "hamur", "patnos", "taşlıçay", "tutak"]},
    "aksaray":         {"official": "Aksaray",         "districts": ["merkez", "ağaçören", "eskil", "gülağaç", "güzelyurt", "ortaköy", "sarıyahşi"]},
    "amasya":          {"official": "Amasya",          "districts": ["merkez", "göynücek", "gümüşhacıköy", "hamamözü", "merzifon", "suluova", "taşova"]},
    "ankara":          {"official": "Ankara",          "districts": ["çankaya", "keçiören", "mamak", "yenimahalle", "altındağ", "sincan", "etimesgut", "pursaklar", "gölbaşı", "polatlı", "beypazarı", "nallıhan", "ayaş", "bala", "çamlıdere", "elmadağ", "güdül", "haymana", "kalecik", "kazan", "kızılcahamam", "şereflikoçhisar"]},
    "antalya":         {"official": "Antalya",         "districts": ["muratpaşa", "kepez", "konyaaltı", "aksu", "alanya", "akseki", "demre", "döşemealtı", "elmalı", "finike", "gazipaşa", "gündoğmuş", "ibradı", "kaş", "kemer", "kumluca", "manavgat", "serik"]},
    "artvin":          {"official": "Artvin",          "districts": ["merkez", "ardanuç", "arhavi", "borçka", "hopa", "murgul", "şavşat", "yusufeli"]},
    "aydın":           {"official": "Aydın",           "districts": ["efeler", "didim", "kuşadası", "nazilli", "söke", "çine", "germencik", "incirliova", "karacasu", "karpuzlu", "koçarlı", "köşk", "sultanhisar", "yenipazar"]},
    "balıkesir":       {"official": "Balıkesir",       "districts": ["altıeylül", "karesi", "ayvalık", "bandırma", "bigadiç", "burhaniye", "dursunbey", "edremit", "erdek", "gömeç", "gönen", "havran", "kepsut", "manyas", "marmara", "savaştepe", "sındırgı", "susurluk"]},
    "bilecik":         {"official": "Bilecik",         "districts": ["merkez", "bozüyük", "gölpazarı", "inhisar", "osmaneli", "pazaryeri", "söğüt", "yenipazar"]},
    "bingöl":          {"official": "Bingöl",          "districts": ["merkez", "adaklı", "genç", "karlıova", "kiğı", "solhan", "yayladere", "yedisu"]},
    "bitlis":          {"official": "Bitlis",          "districts": ["merkez", "adilcevaz", "ahlat", "güroymak", "hizan", "mutki", "tatvan"]},
    "bolu":            {"official": "Bolu",            "districts": ["merkez", "dörtdivan", "gerede", "göynük", "kıbrıscık", "mengen", "mudurnu", "seben", "yeniçağa"]},
    "burdur":          {"official": "Burdur",          "districts": ["merkez", "ağlasun", "altınyayla", "bucak", "çavdır", "çeltikçi", "gölhisar", "karamanlı", "kemer", "tefenni", "yeşilova"]},
    "bursa":           {"official": "Bursa",           "districts": ["osmangazi", "nilüfer", "yıldırım", "büyükorhan", "gemlik", "gürsu", "harmancık", "inegöl", "iznik", "karacabey", "keles", "kestel", "mudanya", "mustafakemalpaşa", "orhaneli", "orhangazi", "yenişehir"]},
    "çanakkale":       {"official": "Çanakkale",       "districts": ["merkez", "ayvacık", "bayramiç", "biga", "bozcaada", "çan", "eceabat", "ezine", "gelibolu", "gökçeada", "lapseki", "yenice"]},
    "çankırı":         {"official": "Çankırı",         "districts": ["merkez", "atkaracalar", "bayramören", "çerkeş", "eldivan", "ılgaz", "korgun", "kurşunlu", "orta", "şabanözü", "yapraklı"]},
    "çorum":           {"official": "Çorum",           "districts": ["merkez", "alaca", "bayat", "boğazkale", "dodurga", "iskilip", "kargı", "laçin", "mecitözü", "ortaköy", "osmancık", "sungurlu", "uğurludağ"]},
    "denizli":         {"official": "Denizli",         "districts": ["pamukkale", "merkezefendi", "acıpayam", "babadağ", "baklan", "bekilli", "beyağaç", "bozkurt", "buldan", "çal", "çameli", "çardak", "çivril", "güney", "honaz", "kale", "sarayköy", "serinhisar", "tavas"]},
    "diyarbakır":      {"official": "Diyarbakır",      "districts": ["bağlar", "kayapınar", "sur", "yenişehir", "bismil", "çermik", "çınar", "çüngüş", "dicle", "eğil", "ergani", "hani", "hazro", "kulp", "lice", "silvan"]},
    "düzce":           {"official": "Düzce",           "districts": ["merkez", "akçakoca", "cumayeri", "çilimli", "gölyaka", "gümüşova", "kaynaşlı", "yığılca"]},
    "edirne":          {"official": "Edirne",          "districts": ["merkez", "enez", "havsa", "ipsala", "keşan", "lalapaşa", "meriç", "süloğlu", "uzunköprü"]},
    "elazığ":          {"official": "Elazığ",          "districts": ["merkez", "ağın", "alacakaya", "arıcak", "baskil", "karakoçan", "keban", "kovancılar", "maden", "palu", "sivrice"]},
    "erzincan":        {"official": "Erzincan",        "districts": ["merkez", "çayırlı", "iliç", "kemah", "kemaliye", "otlukbeli", "refahiye", "tercan", "üzümlü"]},
    "erzurum":         {"official": "Erzurum",         "districts": ["yakutiye", "palandöken", "aziziye", "aşkale", "çat", "hınıs", "horasan", "ispir", "karaçoban", "karayazı", "köprüköy", "narman", "oltu", "olur", "pasinler", "pazaryolu", "şenkaya", "tekman", "tortum", "uzundere"]},
    "eskişehir":       {"official": "Eskişehir",       "districts": ["odunpazarı", "tepebaşı", "alpu", "beylikova", "çifteler", "günyüzü", "han", "inönü", "mahmudiye", "mihalgazi", "mihallıççık", "sarıcakaya", "seyitgazi", "sivrihisar"]},
    "gaziantep":       {"official": "Gaziantep",       "aliases": ["antep"], "districts": ["şahinbey", "şehitkamil", "araban", "islahiye", "karkamış", "nizip", "nurdağı", "oğuzeli", "yavuzeli"]},
    "giresun":         {"official": "Giresun",         "districts": ["merkez", "alucra", "bulancak", "çamoluk", "çanakçı", "dereli", "doğankent", "espiye", "eynesil", "görele", "güce", "keşap", "piraziz", "şebinkarahisar", "tirebolu", "yağlıdere"]},
    "gümüşhane":       {"official": "Gümüşhane",       "districts": ["merkez", "kelkit", "köse", "kürtün", "şiran", "torul"]},
    "hakkari":         {"official": "Hakkari",         "districts": ["merkez", "çukurca", "derecik", "şemdinli", "yüksekova"]},
    "hatay":           {"official": "Hatay",           "aliases": ["antakya", "iskenderun"], "districts": ["antakya", "iskenderun", "defne", "arsuz", "payas", "dörtyol", "kırıkhan", "reyhanlı", "samandağ", "altınözü", "belen", "erzin", "hassa", "kumlu", "yayladağı"]},
    "ığdır":           {"official": "Iğdır",           "aliases": ["igdir"], "districts": ["merkez", "aralık", "karakoyunlu", "tuzluca"]},
    "isparta":         {"official": "Isparta",         "districts": ["merkez", "atabey", "eğirdir", "gelendost", "gönen", "keçiborlu", "senirkent", "sütçüler", "şarkikaraağaç", "uluborlu", "yalvaç", "yenişarbademli"]},
    "istanbul":        {"official": "İstanbul",        "aliases": ["istanbul"], "districts": ["adalar", "arnavutköy", "ataşehir", "avcılar", "bağcılar", "bahçelievler", "bakırköy", "başakşehir", "bayrampaşa", "beşiktaş", "beykoz", "beylikdüzü", "beyoğlu", "büyükçekmece", "çatalca", "çekmeköy", "esenler", "esenyurt", "eyüpsultan", "fatih", "gaziosmanpaşa", "güngören", "kadıköy", "kağıthane", "kartal", "küçükçekmece", "maltepe", "pendik", "sancaktepe", "sarıyer", "silivri", "sultanbeyli", "sultangazi", "şile", "şişli", "tuzla", "ümraniye", "üsküdar", "zeytinburnu"]},
    "izmir":           {"official": "İzmir",           "aliases": ["izmir", "smyrna"], "districts": ["konak", "karşıyaka", "bornova", "buca", "çiğli", "gaziemir", "güzelbahçe", "karabağlar", "bayraklı", "balçova", "narlıdere", "aliağa", "bayındır", "bergama", "beydağ", "çeşme", "dikili", "foça", "karaburun", "kemalpaşa", "kınık", "kiraz", "menderes", "menemen", "ödemiş", "seferihisar", "selçuk", "tire", "torbalı", "urla"]},
    "kahramanmaraş":   {"official": "Kahramanmaraş",   "aliases": ["maraş"], "districts": ["dulkadiroğlu", "onikişubat", "afşin", "andırın", "çağlayancerit", "ekinözü", "elbistan", "göksun", "nurhak", "pazarcık", "türkoğlu"]},
    "karabük":         {"official": "Karabük",         "districts": ["merkez", "eflani", "eskipazar", "ovacık", "safranbolu", "yenice"]},
    "karaman":         {"official": "Karaman",         "districts": ["merkez", "ayrancı", "başyayla", "ermenek", "kazımkarabekir", "sarıveliler"]},
    "kars":            {"official": "Kars",            "districts": ["merkez", "akyaka", "arpaçay", "digor", "kağızman", "sarıkamış", "selim", "susuz"]},
    "kastamonu":       {"official": "Kastamonu",       "districts": ["merkez", "abana", "ağlı", "araç", "azdavay", "bozkurt", "cide", "çatalzeytin", "daday", "devrekani", "doğanyurt", "hanönü", "ihsangazi", "inebolu", "küre", "pınarbaşı", "şenpazar", "taşköprü", "tosya"]},
    "kayseri":         {"official": "Kayseri",         "districts": ["kocasinan", "melikgazi", "talas", "akkışla", "bünyan", "develi", "felahiye", "hacılar", "incesu", "özvatan", "pınarbaşı", "sarıoğlan", "sarız", "tomarza", "yahyalı", "yeşilhisar"]},
    "kilis":           {"official": "Kilis",           "districts": ["merkez", "elbeyli", "musabeyli", "polateli"]},
    "kırıkkale":       {"official": "Kırıkkale",       "districts": ["merkez", "bahşili", "balışeyh", "çelebi", "delice", "karakeçili", "keskin", "sulakyurt", "yahşihan"]},
    "kırklareli":      {"official": "Kırklareli",      "districts": ["merkez", "babaeski", "demirköy", "kofçaz", "lüleburgaz", "pehlivanköy", "pınarhisar", "vize"]},
    "kırşehir":        {"official": "Kırşehir",        "districts": ["merkez", "akçakent", "akpınar", "boztepe", "çiçekdağı", "kaman", "mucur"]},
    "kocaeli":         {"official": "Kocaeli",         "aliases": ["izmit"], "districts": ["izmit", "başiskele", "çayırova", "darıca", "derince", "dilovası", "gebze", "gölcük", "kandıra", "karamürsel", "kartepe", "körfez"]},
    "konya":           {"official": "Konya",           "districts": ["selçuklu", "karatay", "meram", "ahırlı", "akören", "akşehir", "altınekin", "beyşehir", "bozkır", "cihanbeyli", "çeltik", "çumra", "derbent", "derebucak", "doğanhisar", "emirgazi", "ereğli", "güneysınır", "hadim", "halkapınar", "hüyük", "ilgın", "kadınhanı", "karapınar", "kulu", "sarayönü", "seydişehir", "taşkent", "tuzlukçu", "yalıhüyük", "yunak"]},
    "kütahya":         {"official": "Kütahya",         "districts": ["merkez", "altıntaş", "aslanapa", "çavdarhisar", "domaniç", "dumlupınar", "emet", "gediz", "hisarcık", "pazarlar", "simav", "şaphane", "tavşanlı"]},
    "malatya":         {"official": "Malatya",         "districts": ["battalgazi", "yeşilyurt", "akçadağ", "arapgir", "arguvan", "darende", "doğanşehir", "doğanyol", "hekimhan", "kale", "kuluncak", "pütürge", "yazıhan"]},
    "manisa":          {"official": "Manisa",          "districts": ["şehzadeler", "yunusemre", "ahmetli", "akhisar", "alaşehir", "demirci", "gölmarmara", "gördes", "kırkağaç", "köprübaşı", "kula", "salihli", "sarıgöl", "saruhanlı", "selendi", "soma", "turgutlu"]},
    "mardin":          {"official": "Mardin",          "districts": ["artuklu", "derik", "dargeçit", "kızıltepe", "mazıdağı", "midyat", "nusaybin", "ömerli", "savur", "yeşilli"]},
    "mersin":          {"official": "Mersin",          "aliases": ["içel"], "districts": ["akdeniz", "mezitli", "toroslar", "yenişehir", "anamur", "aydıncık", "bozyazı", "çamlıyayla", "erdemli", "gülnar", "mut", "silifke", "tarsus"]},
    "muğla":           {"official": "Muğla",           "districts": ["menteşe", "bodrum", "dalaman", "datça", "fethiye", "kavaklıdere", "köyceğiz", "marmaris", "milas", "ortaca", "seydikemer", "ula", "yatağan"]},
    "muş":             {"official": "Muş",             "districts": ["merkez", "bulanık", "hasköy", "korkut", "malazgirt", "varto"]},
    "nevşehir":        {"official": "Nevşehir",        "aliases": ["kapadokya", "cappadocia"], "districts": ["merkez", "acıgöl", "avanos", "derinkuyu", "gülşehir", "hacıbektaş", "kozaklı", "ürgüp"]},
    "niğde":           {"official": "Niğde",           "districts": ["merkez", "altunhisar", "bor", "çamardı", "çiftlik", "ulukışla"]},
    "ordu":            {"official": "Ordu",            "districts": ["altınordu", "akkuş", "aybastı", "çamaş", "çatalpınar", "çaybaşı", "fatsa", "gölköy", "gülyalı", "gürgentepe", "ikizce", "kabadüz", "kabataş", "korgan", "kumru", "mesudiye", "perşembe", "ulubey", "ünye"]},
    "osmaniye":        {"official": "Osmaniye",        "districts": ["merkez", "bahçe", "düziçi", "hasanbeyli", "kadirli", "sumbas", "toprakkale"]},
    "rize":            {"official": "Rize",            "districts": ["merkez", "ardeşen", "çamlıhemşin", "çayeli", "derepazarı", "fındıklı", "güneysu", "hemşin", "ikizdere", "iyidere", "kalkandere", "pazar"]},
    "sakarya":         {"official": "Sakarya",         "aliases": ["adapazarı"], "districts": ["adapazarı", "akyazı", "arifiye", "erenler", "ferizli", "geyve", "hendek", "karapürçek", "karasu", "kaynarca", "kocaali", "pamukova", "serdivan", "söğütlü", "taraklı"]},
    "samsun":          {"official": "Samsun",          "districts": ["atakum", "canik", "ilkadım", "tekkeköy", "alaçam", "asarcık", "ayvacık", "bafra", "çarşamba", "havza", "kavak", "ladik", "salıpazarı", "terme", "vezirköprü", "yakakent"]},
    "şanlıurfa":       {"official": "Şanlıurfa",       "aliases": ["urfa"], "districts": ["eyyübiye", "haliliye", "karaköprü", "akçakale", "birecik", "bozova", "ceylanpınar", "halfeti", "harran", "hilvan", "siverek", "suruç", "viranşehir"]},
    "siirt":           {"official": "Siirt",           "districts": ["merkez", "baykan", "eruh", "kurtalan", "pervari", "şirvan", "tillo"]},
    "sinop":           {"official": "Sinop",           "districts": ["merkez", "ayancık", "boyabat", "dikmen", "durağan", "erfelek", "gerze", "saraydüzü", "türkeli"]},
    "şırnak":          {"official": "Şırnak",          "districts": ["merkez", "beytüşşebap", "cizre", "güçlükonak", "idil", "silopi", "uludere"]},
    "sivas":           {"official": "Sivas",           "districts": ["merkez", "akıncılar", "altınyayla", "divriği", "doğanşar", "gemerek", "gölova", "hafik", "imranlı", "kangal", "koyulhisar", "suşehri", "şarkışla", "ulaş", "yıldızeli", "zara"]},
    "tekirdağ":        {"official": "Tekirdağ",        "districts": ["süleymanpaşa", "ergene", "kapaklı", "çerkezköy", "çorlu", "hayrabolu", "malkara", "marmaraereğlisi", "muratlı", "saray", "şarköy"]},
    "tokat":           {"official": "Tokat",           "districts": ["merkez", "almus", "artova", "başçiftlik", "erbaa", "niksar", "pazar", "reşadiye", "sulusaray", "turhal", "yeşilyurt", "zile"]},
    "trabzon":         {"official": "Trabzon",         "districts": ["ortahisar", "akçaabat", "araklı", "arsin", "beşikdüzü", "çarşıbaşı", "çaykara", "dernekpazarı", "düzköy", "hayrat", "köprübaşı", "maçka", "of", "sürmene", "şalpazarı", "tonya", "vakfıkebir", "yomra"]},
    "tunceli":         {"official": "Tunceli",         "aliases": ["dersim"], "districts": ["merkez", "çemişgezek", "hozat", "mazgirt", "nazımiye", "ovacık", "pertek", "pülümür"]},
    "uşak":            {"official": "Uşak",            "districts": ["merkez", "banaz", "eşme", "karahallı", "sivaslı", "ulubey"]},
    "van":             {"official": "Van",             "districts": ["ipekyolu", "tuşba", "edremit", "bahçesaray", "başkale", "çaldıran", "çatak", "erciş", "gevaş", "gürpınar", "muradiye", "özalp", "saray"]},
    "yalova":          {"official": "Yalova",          "districts": ["merkez", "altınova", "armutlu", "çiftlikköy", "çınarcık", "termal"]},
    "yozgat":          {"official": "Yozgat",          "districts": ["merkez", "akdağmadeni", "aydıncık", "boğazlıyan", "çandır", "çayıralan", "çekerek", "kadışehri", "saraykent", "sarıkaya", "şefaatli", "sorgun", "yenifakılı", "yerköy"]},
    "zonguldak":       {"official": "Zonguldak",       "districts": ["merkez", "alaplı", "çaycuma", "devrek", "ereğli", "gökçebey", "kilimli", "kozlu"]},
}

PROVINCE_CODES: Dict[str, str] = {
    "Adana": "TR-01", "Adıyaman": "TR-02", "Afyonkarahisar": "TR-03",
    "Ağrı": "TR-04", "Aksaray": "TR-68", "Amasya": "TR-05",
    "Ankara": "TR-06", "Antalya": "TR-07", "Ardahan": "TR-75",
    "Artvin": "TR-08", "Aydın": "TR-09", "Balıkesir": "TR-10",
    "Bartın": "TR-74", "Batman": "TR-72", "Bayburt": "TR-69",
    "Bilecik": "TR-11", "Bingöl": "TR-12", "Bitlis": "TR-13",
    "Bolu": "TR-14", "Burdur": "TR-15", "Bursa": "TR-16",
    "Çanakkale": "TR-17", "Çankırı": "TR-18", "Çorum": "TR-19",
    "Denizli": "TR-20", "Diyarbakır": "TR-21", "Düzce": "TR-81",
    "Edirne": "TR-22", "Elazığ": "TR-23", "Erzincan": "TR-24",
    "Erzurum": "TR-25", "Eskişehir": "TR-26", "Gaziantep": "TR-27",
    "Giresun": "TR-28", "Gümüşhane": "TR-29", "Hakkari": "TR-30",
    "Hatay": "TR-31", "Iğdır": "TR-76", "Isparta": "TR-32",
    "İstanbul": "TR-34", "İzmir": "TR-35", "Kahramanmaraş": "TR-46",
    "Karabük": "TR-78", "Karaman": "TR-70", "Kars": "TR-36",
    "Kastamonu": "TR-37", "Kayseri": "TR-38", "Kilis": "TR-79",
    "Kırıkkale": "TR-71", "Kırklareli": "TR-39", "Kırşehir": "TR-40",
    "Kocaeli": "TR-41", "Konya": "TR-42", "Kütahya": "TR-43",
    "Malatya": "TR-44", "Manisa": "TR-45", "Mardin": "TR-47",
    "Mersin": "TR-33", "Muğla": "TR-48", "Muş": "TR-49",
    "Nevşehir": "TR-50", "Niğde": "TR-51", "Ordu": "TR-52",
    "Osmaniye": "TR-80", "Rize": "TR-53", "Sakarya": "TR-54",
    "Samsun": "TR-55", "Şanlıurfa": "TR-63", "Siirt": "TR-56",
    "Sinop": "TR-57", "Şırnak": "TR-73", "Sivas": "TR-58",
    "Tekirdağ": "TR-59", "Tokat": "TR-60", "Trabzon": "TR-61",
    "Tunceli": "TR-62", "Uşak": "TR-64", "Van": "TR-65",
    "Yalova": "TR-77", "Yozgat": "TR-66", "Zonguldak": "TR-67",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOCATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_lookup(locations: dict) -> dict:
    lookup = {}
    for key, val in locations.items():
        official_province = val["official"]
        lookup[key] = (official_province, "")
        for alias in val.get("aliases", []):
            lookup[alias.lower()] = (official_province, "")
        for district in val.get("districts", []):
            d_lower = district.lower()
            if d_lower not in lookup:
                lookup[d_lower] = (official_province, district.title())
    return lookup

LOCATION_LOOKUP = _build_lookup(TURKEY_LOCATIONS)
SORTED_CANDIDATES = sorted(LOCATION_LOOKUP.keys(), key=len, reverse=True)

COMPILED_PATTERNS = {
    candidate: re.compile(r'(?<![a-zA-ZğüşıöçĞÜŞİÖÇ])' + re.escape(candidate) + r'(?![a-zA-ZğüşıöçĞÜŞİÖÇ])')
    for candidate in SORTED_CANDIDATES
}

def extract_location(text: str, post_tags: list = None) -> Tuple[str, str]:
    def scan(source: str) -> Tuple[str, str]:
        if not source:
            return "", ""
        normalized = source.lower()
        best_province = ""
        for candidate in SORTED_CANDIDATES:
            if COMPILED_PATTERNS[candidate].search(normalized):
                province, district = LOCATION_LOOKUP[candidate]
                if district:
                    return province, district
                elif not best_province:
                    best_province = province
        return best_province, ""

    for tag in (post_tags or []):
        province, district = scan(str(tag))
        if province:
            return province, district

    return scan(text)

def tag_posts_with_location(posts_col, limit: int = 0) -> int:
    query = {
        "$or": [
            {"location.province": {"$exists": False}},
            {"location.province": None},
            {"location.province": ""},
        ]
    }
    cursor = posts_col.find(query) if limit == 0 else posts_col.find(query).limit(limit)
    bulk_ops = []

    for post in cursor:
        text = post.get("text", "")
        post_tags = post.get("post_tags", [])
        province, district = extract_location(text, post_tags)

        bulk_ops.append(UpdateOne(
            {"_id": post["_id"]},
            {"$set": {"location.province": province, "location.district": district}}
        ))

    if bulk_ops:
        res = posts_col.bulk_write(bulk_ops)
        print(f"  Location tagging: {res.modified_count} posts updated.")
    return len(bulk_ops)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FALLBACK RULES & MAP SCORING
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_PROVINCE = "Unknown"
FALLBACK_DISTRICT = "Unknown"

def apply_fallback(province: str, district: str) -> Tuple[str, str]:
    province = (province or "").strip()
    district = (district or "").strip()
    if not province:
        return FALLBACK_PROVINCE, FALLBACK_DISTRICT
    if not district:
        return province, province
    return province, district

def map_color_score(positive: int, neutral: int, negative: int) -> float:
    total = positive + neutral + negative
    if total == 0:
        return 0.5
    raw = (positive - negative) / total
    return round((raw + 1) / 2, 4)

def sentiment_label_from_score(score: float) -> str:
    if score >= 0.6: return "positive"
    if score <= 0.4: return "negative"
    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEOJSON FEATURE BUILDERS (UPDATED FOR MULTI-MODEL)
# ─────────────────────────────────────────────────────────────────────────────

def empty_model_stats() -> Dict[str, Any]:
    return {
        "scores": [], 
        "positive": 0, 
        "neutral": 0, 
        "negative": 0,
        "categories": {}
    }

def empty_bucket() -> Dict[str, Any]:
    return {
        "llm": empty_model_stats(),
        "transformer": empty_model_stats()
    }

def _update_stats(stats: Dict, label: str, score: Any, category: str) -> None:
    label = (label or "neutral").lower().strip()
    category = (category or "uncategorized").lower().strip()
    
    # Update main totals
    if label == "positive": stats["positive"] += 1
    elif label == "negative": stats["negative"] += 1
    else: stats["neutral"] += 1
    
    # Initialize category dictionary if it doesn't exist yet
    if category not in stats["categories"]:
        stats["categories"][category] = {"positive": 0, "neutral": 0, "negative": 0}
        
    # Update category totals
    if label == "positive": stats["categories"][category]["positive"] += 1
    elif label == "negative": stats["categories"][category]["negative"] += 1
    else: stats["categories"][category]["neutral"] += 1
    
    if isinstance(score, (int, float)):
        stats["scores"].append(score)

def add_post_to_bucket(bucket: Dict, sentiment: Dict, category: str) -> None:
    # Process LLM Sentiment
    llm_sent = sentiment.get("llm", {})
    _update_stats(bucket["llm"], llm_sent.get("label"), llm_sent.get("score"), category)
    
    # Process Transformer Sentiment
    trans_sent = sentiment.get("transformer", {})
    _update_stats(bucket["transformer"], trans_sent.get("label"), trans_sent.get("score"), category)

def _compute_model_stats(stats: Dict) -> Dict[str, Any]:
    pos, neu, neg = stats["positive"], stats["neutral"], stats["negative"]
    total = pos + neu + neg
    color = map_color_score(pos, neu, neg)
    
    # Format the category data for the frontend GeoJSON
    category_breakdown = {}
    for cat, counts in stats["categories"].items():
        c_pos, c_neu, c_neg = counts["positive"], counts["neutral"], counts["negative"]
        category_breakdown[cat] = {
            "total": c_pos + c_neu + c_neg,
            "map_color_score": map_color_score(c_pos, c_neu, c_neg)
        }
        
    return {
        "total_posts": total,
        "distribution": {"positive": pos, "neutral": neu, "negative": neg},
        "map_color_score": color,
        "sentiment_label": sentiment_label_from_score(color),
        "categories": category_breakdown
    }

def compute_stats(bucket: Dict) -> Dict[str, Any]:
    return {
        "llm": _compute_model_stats(bucket["llm"]),
        "transformer": _compute_model_stats(bucket["transformer"])
    }

def build_feature(level: str, province: str, stats: Dict, district: str = None) -> Dict:
    props = {
        "level": level,
        "province": province,
        "province_code": PROVINCE_CODES.get(province, "TR-??"),
        "models": stats  # Nest the llm and transformer stats here
    }
    if district:
        props["district"] = district
        
    return {
        "type": "Feature",
        "geometry": None,
        "properties": props
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PIPELINE ENTRY POINT 
# ─────────────────────────────────────────────────────────────────────────────

def run(input_data: Any, context: Dict[str, Any]) -> Dict:
    db = context["db"]
    posts_col = db["posts_processed"]
    tw_end = datetime.now(timezone.utc)

    print("  [Step 1] Tagging untagged posts with province / district ...")
    tag_posts_with_location(posts_col)

    print("  [Step 2] Grouping data for GeoJSON output (Multi-Model + Category logic)...")
    
# AMMAR'S HOTFIX: Join via _id, but extract the category from Yusuf's post_tags array
    pipeline = [
        {
            "$lookup": {
                "from": "posts_raw",
                "localField": "post_id",
                "foreignField": "_id",
                "as": "raw_doc"
            }
        },
        {
            "$unwind": {
                "path": "$raw_doc",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$project": {
                "location": 1,
                "sentiment": 1,
                # Grab the first element (index 0) of the post_tags array
                "category": { "$arrayElemAt": ["$raw_doc.post_tags", 0] } 
            }
        }
    ]
    
    cursor = posts_col.aggregate(pipeline)

    province_buckets: Dict[str, Dict] = defaultdict(empty_bucket)
    district_buckets: Dict[Tuple, Dict] = defaultdict(empty_bucket)
    unknown_count = 0

    for post in cursor:
        loc = post.get("location") or {}
        sent = post.get("sentiment") or {}
        cat = post.get("category", "uncategorized") 
        
        province, district = apply_fallback(loc.get("province", ""), loc.get("district", ""))
        if province == FALLBACK_PROVINCE: unknown_count += 1

        add_post_to_bucket(province_buckets[province], sent, cat)
        add_post_to_bucket(district_buckets[(province, district)], sent, cat)

    all_features = []
    for prov, bucket in province_buckets.items():
        all_features.append(build_feature("province", prov, compute_stats(bucket)))
        
    for (prov, dist), bucket in district_buckets.items():
        all_features.append(build_feature("district", prov, compute_stats(bucket), dist))

    print("  [Step 3] Building GeoJSON FeatureCollection ...")
    return {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": tw_end.isoformat(),
            "total_provinces": len(province_buckets),
            "total_districts": len(district_buckets),
            "unmatched_posts": unknown_count,
        },
        "features": all_features,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6.  STANDALONE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pymongo import MongoClient
    import os

    MONGO_URI = os.getenv("MONGO_URI",
        "mongodb+srv://COM6064:OnTHCZcqye91Yv1s@cluster0.bj4tnnh.mongodb.net/COM6064?appName=Cluster0")
    MONGO_DB  = os.getenv("MONGO_DB_NAME", "COM6064")

    client  = MongoClient(MONGO_URI)
    context = {"db": client[MONGO_DB]}

    print("=" * 60)
    print("  geospatial.py — standalone run")
    print("=" * 60)

    result = run(None, context)
    client.close()

    print("\n" + "=" * 60)
    print("  GeoJSON OUTPUT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
