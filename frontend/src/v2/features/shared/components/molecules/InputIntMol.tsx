import { BorderBoxAtm } from "@/v2/features/shared/components/atoms/BorderBoxAtm";

interface Props {
  value: number;
  min?: number;
  max?: number;
  onChange?: (_: number) => void;
  disabled?: boolean;
  className?: string;
}

export function InputIntMol({
  value,
  min,
  max,
  onChange,
  disabled,
  className,
}: Props) {
  const parseIntMinMax = (value: string) => {
    let valueInt = parseInt(value, 10);
    if (!isNaN(valueInt)) {
      valueInt = Math.max(min ?? valueInt, valueInt);
      valueInt = Math.min(max ?? valueInt, valueInt);
      return valueInt;
    }
    return undefined;
  };

  return (
    <BorderBoxAtm
      className={
        "flex h-8 items-center justify-center" +
        (className ? ` ${className}` : "")
      }
    >
      <input
        type="text"
        pattern="^-?[0-9]*$"
        value={value}
        onChange={(event) => {
          if (onChange) {
            const onChangeValue = parseIntMinMax(event.target.value);
            onChangeValue && onChange(onChangeValue);
          }
        }}
        disabled={disabled}
        className="hide-spinner w-full bg-transparent text-center outline-none"
      />
    </BorderBoxAtm>
  );
}
