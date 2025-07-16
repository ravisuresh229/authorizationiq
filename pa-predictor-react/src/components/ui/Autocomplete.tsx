import React, { useState, useEffect, useRef } from 'react';
import { debounce } from 'lodash';

interface AutocompleteOption {
  value: string;
  label: string;
}

interface AutocompleteProps {
  options: AutocompleteOption[];
  onSelect: (selected: AutocompleteOption[]) => void;
  placeholder: string;
  multiple?: boolean;
}

const Autocomplete: React.FC<AutocompleteProps> = ({
  options,
  onSelect,
  placeholder,
  multiple = false
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState(options);
  const [selectedItems, setSelectedItems] = useState<AutocompleteOption[]>([]);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const debouncedSearch = debounce((term: string) => {
    const filtered = options.filter(option =>
      option.value.toLowerCase().includes(term.toLowerCase()) ||
      option.label.toLowerCase().includes(term.toLowerCase())
    );
    setFilteredOptions(filtered.slice(0, 8));
  }, 200);

  useEffect(() => {
    debouncedSearch(searchTerm);
  }, [searchTerm, options, debouncedSearch]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setSearchTerm(newValue);
    setIsOpen(true);
  };

  const handleOptionSelect = (option: AutocompleteOption) => {
    if (multiple) {
      const isAlreadySelected = selectedItems.some(item => item.value === option.value);
      let newSelectedItems: AutocompleteOption[];
      
      if (isAlreadySelected) {
        newSelectedItems = selectedItems.filter(item => item.value !== option.value);
      } else {
        newSelectedItems = [...selectedItems, option];
      }
      
      setSelectedItems(newSelectedItems);
      onSelect(newSelectedItems);
    } else {
      onSelect([option]);
      setSearchTerm(option.label);
      setIsOpen(false);
    }
  };

  const removeSelectedItem = (itemToRemove: AutocompleteOption) => {
    const newSelectedItems = selectedItems.filter(item => item.value !== itemToRemove.value);
    setSelectedItems(newSelectedItems);
    onSelect(newSelectedItems);
  };

  const isOptionSelected = (option: AutocompleteOption) => {
    return selectedItems.some(item => item.value === option.value);
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <div className="relative">
        {multiple && selectedItems.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {selectedItems.map((item) => (
              <span
                key={item.value}
                className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 hover:bg-gray-200 transition-all"
              >
                {item.label}
                <button
                  type="button"
                  className="ml-2 text-gray-400 hover:text-gray-600"
                  onClick={() => removeSelectedItem(item)}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        
        <div className="relative">
          <span className="absolute left-0 top-1/2 transform -translate-y-1/2 text-gray-400">🔍</span>
          <input
            type="text"
            value={searchTerm}
            onChange={handleInputChange}
            onFocus={() => setIsOpen(true)}
            placeholder={placeholder}
            className="w-full pl-6 pr-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light bg-transparent transition-all"
          />
        </div>
      </div>

      {isOpen && filteredOptions.length > 0 && (
        <div className="absolute z-50 w-full mt-2 bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="max-h-72 overflow-auto">
            {filteredOptions.map((option, index) => (
              <div
                key={index}
                className={`px-4 py-3 cursor-pointer hover:bg-gray-50 transition-all ${
                  isOptionSelected(option) ? 'bg-gray-50' : ''
                }`}
                onClick={() => handleOptionSelect(option)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900">{option.value}</div>
                    <div className="text-sm text-gray-500 mt-0.5">{option.label}</div>
                  </div>
                  {isOptionSelected(option) && (
                    <span className="text-black">✓</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Autocomplete; 