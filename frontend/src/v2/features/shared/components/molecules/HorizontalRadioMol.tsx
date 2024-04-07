import { BorderBoxAtm } from "@/v2/features/shared/components/atoms/BorderBoxAtm";
import { RadioGroup } from "@headlessui/react";

interface Props<T extends string | number> {
  possibleValues: T[];
  value: T;
  setValue: (_: T) => void;
  className?: string;
}

export function HorizontalRadioMol<T extends string | number>({
  possibleValues,
  value,
  setValue,
  className,
}: Props<T>) {
  return (
    <RadioGroup value={value} onChange={setValue}>
      <BorderBoxAtm
        className={
          "grid h-8 grid-flow-col justify-stretch divide-x text-xs" +
          (className ? ` ${className}` : "")
        }
      >
        {possibleValues.map((option, idx) => (
          <RadioGroup.Option value={option} key={idx} className="outline-none">
            {({ checked }) => (
              <label
                className={
                  "flex h-full cursor-pointer items-center justify-center" +
                  (checked ? " bg-white text-black" : "") +
                  (idx == 0 ? " rounded-l" : "") +
                  (idx == possibleValues.length - 1 ? " rounded-r" : "")
                }
              >
                {option}
              </label>
            )}
          </RadioGroup.Option>
        ))}
      </BorderBoxAtm>
    </RadioGroup>
  );
}
