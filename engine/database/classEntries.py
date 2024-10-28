
from helper.text_decorators import convert_english_to_persian, split_string_language_specific




class Entries:

    def __init__(self, platePercent, charPercent, eDate, eTime, plateNum, status):
        self.status = status
        self.plateNum = plateNum
        self.eTime = eTime
        self.eDate = eDate

        self.charPercent = charPercent
        self.platePercent = platePercent

    def getTime(self):
        return self.eTime

    def getDate(self, persian=True):
       
        return self.eDate

    def getPlatePic(self):
        date=self.eDate.split('-')
        date=f"{date[1]}-{date[2]}"
        
        
        return 'temp/{}_{}.jpg'.format(self.plateNum, date)

    def getCharPercent(self):
        return "{}%".format(self.charPercent)

    def getPlatePercent(self):
        return "{}%".format(self.platePercent)

    def getPlateNumber(self, display=False):
        return convert_english_to_persian(split_string_language_specific(self.plateNum), display)


