import { BorderBox } from "../atoms/BorderBox";

interface Props {
  value: number;
  onChange: (_: number) => void;
  min?: number;
  max?: number;
  className?: string;
}

export function InputNumber({ value, onChange, min, max, className }: Props) {
  return (
    <BorderBox
      className={
        "flex h-8 items-center justify-center" +
        (className ? ` ${className}` : "")
      }
    >
      <input
        type="text"
        pattern="[0-9]*"
        value={value}
        onChange={(event) => {
          let onChangeValue = parseInt(event.target.value, 10);
          if (!isNaN(onChangeValue)) {
            onChangeValue = Math.max(min ?? onChangeValue, onChangeValue);
            onChangeValue = Math.min(max ?? onChangeValue, onChangeValue);
            onChange(onChangeValue);
          }
        }}
        className="hide-spinner w-full bg-transparent text-center outline-none"
      />
    </BorderBox>
  );
}
