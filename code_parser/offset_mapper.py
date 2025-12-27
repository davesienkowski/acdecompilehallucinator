from bisect import bisect_right

class OffsetMapper:
    def __init__(self, filepath):
        """Load and parse the lines.txt file."""
        self.address_map = []  # List of (addr, filename, line_num)
        self.found_files = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        
        for line in lines:
            if "line/addr" in line:
                # Parse the header line
                # Format: "  d:\path\to\file.cpp (None), 0001:002C1420-002C1452, line/addr pairs = 3"
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    # Extract filename
                    filename = parts[0].split('(')[0].strip()
                    current_file = filename
            
            elif current_file and line.strip() and not line.strip().startswith('**'):
                # This is a line with line_num/address pairs
                # Format: "742 002C1420    743 002C1420    744 002C144F"
                parts = line.strip().split()
                i = 0
                while i < len(parts) - 1:
                    try:
                        line_num = parts[i]
                        addr_hex = parts[i + 1]
                        addr = int(addr_hex, 16)
                        
                        # Store (address, filename, line_number)
                        self.address_map.append((addr, current_file, line_num))
                        i += 2
                    except (ValueError, IndexError):
                        i += 1
        
        # Sort by address for binary search
        self.address_map.sort(key=lambda x: x[0])
        
    def get_filename(self, offset):
        """
        Given an offset (e.g., 0x0055A460), return the filename with line number.
        Args:
            offset: int or hex string (e.g., 0x0055A460 or '0x0055A460')
        Returns:
            str: "filename:line_number address" if found, None otherwise
        """
        # Convert offset to int if it's a string
        if isinstance(offset, str):
            offset = int(offset, 16)
        
        # Subtract base address
        adjusted = offset - 0x0401000
        
        # Binary search for exact match or closest address
        idx = bisect_right(self.address_map, adjusted, key=lambda x: x[0])
        
        # Check for exact match first
        if idx > 0:
            addr, filename, line_num = self.address_map[idx - 1]
            if addr == adjusted:
                return f"{filename}:{line_num} {adjusted:08X}"
        
        # If no exact match, find the closest address before this one
        # (the function that contains this offset)
        if idx > 0:
            addr, filename, line_num = self.address_map[idx - 1]
        
            if filename not in self.found_files:
                self.found_files.append(filename)
                
            # Return the last known line before this address
            return f"{filename}:{line_num} {adjusted:08X}"

        return None