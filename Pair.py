class Pair:
    def __init__(self, row, column):
        self.row = row
        self.column = column

    def getRow(self):
        return self.row

    def getColumn(self):
        return self.column

    def __hash__(self):
        prime = 31
        c = max(self.column, self.row)
        r = self.column + self.row - c
        result = prime * c + r
        return result

    def __eq__(self, other):
        if isinstance(other, Pair):
            return (self.column == other.column and self.row == other.row) or \
                   (self.column == other.row and self.row == other.column)
        return False

    def __str__(self):
        return "Pair [row=" + str(self.row) + ", column=" + str(self.column) + "]"
